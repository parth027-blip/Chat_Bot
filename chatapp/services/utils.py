from sentence_transformers import SentenceTransformer
from chatapp.models import User, Category, Service, ProfileDailyVisit
from chatapp.services.ai import ask_llama
from django.db import connection

import json
import re
import logging

from pinecone import Pinecone
from django.conf import settings

logger = logging.getLogger(__name__)

model = SentenceTransformer("all-MiniLM-L6-v2")

# ─────────────────────────────────────────────
# PINECONE SETUP
# ─────────────────────────────────────────────
pc = Pinecone(api_key=settings.PINECONE_API_KEY)
pinecone_index = pc.Index(settings.PINECONE_INDEX_NAME)

# ─────────────────────────────────────────────
# DATABASE SCHEMA CONTEXT FOR LLAMA
# ─────────────────────────────────────────────
DB_SCHEMA = """
You have access to a MySQL database with these tables:

1. users
   - id, name, email, phone, business_name
   - category_id (FK → categories.id)
   - city (FK → cities.id)
   - state (int), pincode, address
   - description, website
   - latitude, longitude
   - is_active (1=active), is_verified (1=verified)
   - created_at, deleted_at

2. categories
   - id, name, description, keywords
   - is_active (1=active)
   - category_type: enum('festival','daily','greeting','motivation','business')

3. cities
   - id, city (name), state_id, is_top (1=top city)

4. services
   - id, service_title, description, keywords
   - user_id (FK → users.id)
   - category_id (FK → categories.id)
   - is_active (1=active)
   - deleted_at (NULL = not deleted)

5. profile_daily_visits
   - id, profile_id (FK → users.id)
   - visit_date (DATE), visits (int)

KEY RULES:
- users.city = cities.id
- users.category_id = categories.id
- Always filter: users.deleted_at IS NULL AND users.is_active = 1
- Always filter: services.deleted_at IS NULL AND services.is_active = 1
"""

ALLOWED_FILTER_FIELDS = {
    "city__city__icontains",
    "category__name__icontains",
    "business_name__icontains",
}

ALLOWED_TABLES = {"users", "categories", "cities", "services", "profile_daily_visits"}
FORBIDDEN_KEYWORDS = {"drop", "delete", "truncate", "insert", "update", "alter", "create", "grant", "revoke"}


# ─────────────────────────────────────────────
# SQL SAFETY
# ─────────────────────────────────────────────
def is_safe_sql(sql: str) -> bool:
    sql_lower = sql.lower()
    if not sql_lower.strip().startswith("select"):
        return False
    for keyword in FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{keyword}\b", sql_lower):
            return False
    return True


# ─────────────────────────────────────────────
# PINECONE SYNC
# ─────────────────────────────────────────────
def sync_businesses_to_pinecone():
    users = (
        User.objects
        .select_related("category", "city")
        .filter(
            is_active=1,
            deleted_at__isnull=True,
            category__isnull=False,
            business_name__isnull=False,
        )
        .exclude(business_name__exact="")
    )

    vectors = []
    for u in users:
        category_name = u.category.name if u.category_id else ""
        city_name     = u.city.city if u.city_id else ""
        description   = u.description or ""
        keywords      = u.category.keywords if u.category and u.category.keywords else ""

        text = f"{u.business_name} is a {category_name} business"
        if city_name:
            text += f" in {city_name}"
        if keywords:
            text += f". Services: {keywords}"
        if description:
            text += f". {description[:150]}"

        embedding = model.encode(text).tolist()

        vectors.append({
            "id": str(u.id),
            "values": embedding,
            "metadata": {
                "business_name": u.business_name or "",
                "category":      category_name or "Unknown",
                "city":          city_name or "—",
                "phone":         u.phone or "",
                "text":          text,
            }
        })

    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        pinecone_index.upsert(vectors=vectors[i:i + batch_size])
        print(f"Upserted {i} to {i + batch_size}")

    print(f"Done. Synced {len(vectors)} businesses.")


# ─────────────────────────────────────────────
# SEMANTIC SEARCH
# ─────────────────────────────────────────────
def semantic_search(user_msg: str, top_k: int = 10, min_score: float = 0.45) -> str:
    try:
        query_embedding = model.encode(user_msg.lower()).tolist()
        results = pinecone_index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
        )

        if not results["matches"]:
            return ""

        lines = []
        for match in results["matches"]:
            if match["score"] < min_score:
                continue

            meta = match["metadata"]

            if not meta.get("business_name"):
                continue
            if meta.get("category") in ("Unknown", "", None):
                continue

            city_display = meta.get("city", "—")
            if city_display in ("Unknown", "", None):
                city_display = "—"

            lines.append(
                f"Business: {meta.get('business_name')} | "
                f"Category: {meta.get('category')} | "
                f"City: {city_display} | "
                f"Phone: {meta.get('phone')}"
            )

        return "\n".join(lines)

    except Exception as e:
        logger.error("Pinecone semantic_search failed: %s", e)
        return ""


# ─────────────────────────────────────────────
# TEXT2SQL — FIX 2: stricter prompt + extract SELECT
# ─────────────────────────────────────────────
def generate_sql(user_msg: str) -> str:
    prompt = f"""
{DB_SCHEMA}

STRICT RULES:
- Return ONLY the raw SQL query — absolutely nothing else.
- No explanation, no markdown, no backticks, no comments.
- No sentences before or after the SQL.
- Start your response directly with SELECT.
- Always use table aliases.
- Always include LIMIT (default 20, max 100).
- Never use DROP, DELETE, UPDATE, INSERT, ALTER.
- For business searches always filter: u.deleted_at IS NULL AND u.is_active = 1
- Use LIKE with % for text searches.
- If question is unclear or not related to DB, return only: NONE

Examples:

User: "show restaurants in Ahmedabad"
SELECT u.business_name, u.phone, cat.name as category, ci.city
FROM users u
JOIN categories cat ON u.category_id = cat.id
JOIN cities ci ON u.city = ci.id
WHERE ci.city LIKE '%Ahmedabad%'
AND cat.name LIKE '%restaurant%'
AND u.deleted_at IS NULL AND u.is_active = 1
LIMIT 20;

User: "top 5 most visited businesses"
SELECT u.business_name, u.phone, SUM(p.visits) as total_visits
FROM profile_daily_visits p
JOIN users u ON p.profile_id = u.id
WHERE u.deleted_at IS NULL AND u.is_active = 1
GROUP BY p.profile_id, u.business_name, u.phone
ORDER BY total_visits DESC
LIMIT 5;

User Question: {user_msg}
"""
    try:
        response = ask_llama(prompt).strip()
        response = re.sub(r"```sql|```", "", response).strip()

        # FIX: Extract only SELECT statement if LLaMA adds explanation
        match = re.search(r"(SELECT\s+.+?)(?:;|$)", response, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip() + ";"

        if response.upper().startswith("NONE") or not response:
            return ""

        return response
    except Exception as e:
        logger.error("generate_sql failed: %s", e)
        return ""


# ─────────────────────────────────────────────
# EXECUTE SQL
# ─────────────────────────────────────────────
def run_sql(sql: str) -> str:
    if not sql or not is_safe_sql(sql):
        logger.warning("Blocked unsafe SQL: %s", sql)
        return ""
    try:
        with connection.cursor() as cursor:
            cursor.execute(sql)
            columns = [col[0] for col in cursor.description]
            rows = cursor.fetchall()

        if not rows:
            return ""

        lines = []
        for row in rows:
            parts = [f"{col}: {val}" for col, val in zip(columns, row) if val is not None]
            lines.append(" | ".join(parts))

        return "\n".join(lines)

    except Exception as e:
        logger.error("run_sql failed: %s | SQL: %s", e, sql)
        return ""


# ─────────────────────────────────────────────
# ORM FALLBACK — FIX 3: better JSON extraction
# ─────────────────────────────────────────────
def generate_query(user_msg: str) -> dict:
    prompt = f"""
Convert the user question into a JSON object with Django ORM filter keys.

STRICT RULES:
- Return ONLY valid JSON — no explanation, no backticks, no markdown.
- Allowed keys:
    city__city__icontains
    category__name__icontains
    business_name__icontains
- Add "limit" key if user asks for specific count.
- If nothing matches return: {{}}

Examples:
User: "restaurants in Ahmedabad" → {{"city__city__icontains": "Ahmedabad", "category__name__icontains": "restaurant"}}
User: "IT services" → {{"category__name__icontains": "IT"}}
User: "top 5 businesses" → {{"limit": 5}}
User: "give me address of ibh" → {{}}

User Question: {user_msg}
JSON:"""
    try:
        response = ask_llama(prompt).strip()

        # Clean markdown if LLaMA adds it
        response = re.sub(r"```json|```", "", response).strip()

        # Extract JSON object if LLaMA adds explanation text
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if match:
            return json.loads(match.group())

        return {}
    except json.JSONDecodeError as e:
        logger.warning("generate_query JSON error: %s | Response: %s", e, response)
        return {}
    except Exception as e:
        logger.error("generate_query failed: %s", e)
        return {}


def get_dynamic_data(user_msg: str) -> str:
    query_dict = generate_query(user_msg)
    try:
        limit = int(query_dict.get("limit", 20))
        limit = max(1, min(limit, 100))
    except (TypeError, ValueError):
        limit = 20

    safe_filter = {
        k: v for k, v in query_dict.items()
        if k in ALLOWED_FILTER_FIELDS
    }

    try:
        users = (
            User.objects
            .select_related("category", "city")
            .filter(
                **safe_filter,
                category__isnull=False,
                city__isnull=False,
                is_active=1,
            )[:limit]
        )
    except Exception as e:
        logger.error("ORM filter failed: %s", e)
        return ""

    if not users:
        return ""

    lines = []
    for u in users:
        category_name = u.category.name if u.category_id else "Unknown"
        city_name     = u.city.city if u.city_id else "Unknown"
        lines.append(
            f"Business: {u.business_name} | Category: {category_name} | "
            f"City: {city_name} | Phone: {u.phone}"
        )
    return "\n".join(lines)


# ─────────────────────────────────────────────
# GREETING / FAREWELL / THANKS / HELP WORDS
# ─────────────────────────────────────────────
GREETING_WORDS = {
    "hi", "hello", "hey", "helo", "hii", "hiii",
    "good morning", "good afternoon", "good evening",
    "good night", "howdy", "greetings", "sup", "what's up",
    "whats up", "namaste", "namaskar", "hi there", "hello there",
}

FAREWELL_WORDS = {
    "bye", "goodbye", "good bye", "see you", "see ya",
    "take care", "later", "cya", "ttyl", "farewell"
}

THANKS_WORDS = {
    "thanks", "thank you", "thank you so much", "thx",
    "ty", "appreciated", "cheers", "dhanyawad", "shukriya"
}

HELP_WORDS = {
    "help", "what can you do", "how to use", "what is this",
    "how does this work", "guide me", "assist me",
    "what do you do", "how can you help"
}

GENERAL_TRIGGERS = [
    "how to", "how do i", "how can i", "how should i",
    "tips for", "tips on", "advice", "suggest", "suggestion",
    "best way", "best ways", "best practice", "best strategies",
    "what is", "what are", "explain", "tell me about",
    "improve", "grow", "increase", "boost", "help me with",
    "why is", "why do", "benefits of", "difference between",
    "how does", "what does", "guide", "strategy", "strategies",
    "marketing", "digital marketing", "social media",
    "get more customers", "attract customers",
]

IBH_KNOWLEDGE = """
You are a smart assistant for Indian Business Hub (IBH) — indianbusinesshub.in

ABOUT IBH:
- Indian Business Hub is a FREE online business directory for Indian businesses
- It helps businesses get discovered by customers across India
- Businesses can list for FREE, get verified, and grow digitally

KEY FEATURES:
1. Free Business Listing — Any business can list on IBH for free
2. Verified Badge — Upload documents to get a verified badge for trust
3. Digital Visiting Card — Share business details on WhatsApp, Instagram
4. Festival & Brand Designs — Auto-create festival promo images with your logo
5. One-Page Website — Get a simple website for your business, no coding needed
6. Easy Customer Contact — Customers can call/message businesses directly
7. IBH Mobile App — Available FREE on Google Play

HOW TO ADD A BUSINESS:
Step 1: Go to indianbusinesshub.in and click 'Add Business'
Step 2: Enter business name, phone, email, address, city, category
Step 3: Set a password and click Finish
- It is completely FREE

HOW TO GET VERIFIED:
Step 1: Complete your business profile fully
Step 2: Upload ID proof, business certificate, GST document
Step 3: IBH team reviews and approves
Step 4: Get your Verified Badge

IBH APP:
- Download: play.google.com/store/apps/details?id=com.app.indianbusinesshub

CONTACT IBH:
- Email: info.indianbusinesshub@gmail.com
- Phone: +91 8000841620
- Address: 209-A, Satva Icon, Vastral, Ahmedabad, Gujarat 382418
"""

# ─────────────────────────────────────────────
# GUIDANCE TRIGGERS — FIX 1: added IBH address triggers
# ─────────────────────────────────────────────
GUIDANCE_TRIGGERS = [
    # platform questions
    "what is ibh", "about ibh", "what is indian business hub",
    "what is this website", "what does ibh do", "tell me about ibh",
    # listing
    "how to add", "add my business", "list my business",
    "how to list", "how to register", "get listed", "join ibh",
    # verification
    "how to get verified", "get verified", "verified badge",
    "verify my business", "verification process",
    # app
    "ibh app", "download app", "install app", "app features",
    "mobile app",
    # benefits
    "why use ibh", "benefits of ibh", "is ibh free",
    # contact & address — FIX 1 additions
    "contact ibh", "ibh contact", "ibh phone", "ibh email",
    "ibh address", "address of ibh", "where is ibh",
    "ibh location", "ibh office", "give me address",
    "what is ibh address", "ibh support",
    # new user
    "i am new", "new user", "how to use ibh", "getting started",
    # search help
    "how to search", "how to find business",
    # tips
    "tips for ibh", "how to grow on ibh", "improve my listing",
]


def is_guidance_question(user_msg: str) -> bool:
    msg = user_msg.lower().strip()
    return any(trigger in msg for trigger in GUIDANCE_TRIGGERS)


def handle_guidance(user_msg: str) -> str:
    system = (
        "You are a helpful assistant for Indian Business Hub (IBH). "
        "Answer using only the IBH knowledge provided. "
        "Be friendly, use emojis, keep answers clear and short. "
        "If unsure, suggest contacting IBH support."
    )
    prompt = f"""
{IBH_KNOWLEDGE}

User Question: {user_msg}
Answer:
"""
    try:
        return ask_llama(prompt, system=system).strip()
    except Exception as e:
        logger.error("handle_guidance failed: %s", e)
        return (
            "I'm here to help with Indian Business Hub! 😊\n"
            "📧 info.indianbusinesshub@gmail.com\n"
            "📞 +91 8000841620"
        )


# ─────────────────────────────────────────────
# FAST INTENT DETECTION
# ─────────────────────────────────────────────
def fast_intent(user_msg: str) -> str | None:
    msg       = user_msg.lower().strip()
    msg_clean = msg.strip("?!., ")

    if msg_clean in GREETING_WORDS: return "greeting"
    if msg_clean in FAREWELL_WORDS: return "farewell"
    if msg_clean in THANKS_WORDS:   return "thanks"
    if msg_clean in HELP_WORDS:     return "help"

    for w in ["hi", "hello", "hey", "good morning", "good afternoon",
              "good evening", "namaste", "namaskar"]:
        if msg_clean == w or msg_clean.startswith(w + " "):
            return "greeting"

    for w in ["bye", "goodbye", "see you", "take care"]:
        if msg_clean.startswith(w):
            return "farewell"

    for w in ["thank", "thanks", "thx", "dhanyawad"]:
        if w in msg_clean:
            return "thanks"

    if is_guidance_question(user_msg):
        return "guidance"

    for trigger in GENERAL_TRIGGERS:
        if msg_clean.startswith(trigger) or f" {trigger} " in f" {msg_clean} ":
            return "general"

    return None


def detect_intent(user_msg: str) -> str:
    quick = fast_intent(user_msg)
    if quick:
        return quick

    prompt = f"""
Classify this message for an Indian business directory chatbot into one intent:
- greeting, farewell, thanks, help, guidance, search, general, unknown

guidance = questions about IBH platform (adding business, verification, app, benefits, contact, address, location)
search   = looking for a business or service
general  = business advice or tips

Return ONLY the intent word.

Message: {user_msg}
Intent:
"""
    try:
        response = ask_llama(prompt)
        intent = response.strip().lower().split()[0].strip(".,!?")
        if intent not in {"greeting", "farewell", "thanks", "help",
                          "guidance", "search", "general", "unknown"}:
            return "unknown"
        return intent
    except Exception as e:
        logger.error("detect_intent failed: %s", e)
        return "unknown"


# ─────────────────────────────────────────────
# CONVERSATIONAL HANDLER
# ─────────────────────────────────────────────
def handle_conversational(intent: str, user_msg: str) -> str:
    COMPANY_NAME = "Indian Business Hub"
    system = (
        f"You are a friendly assistant for {COMPANY_NAME}. "
        "Use emojis. Keep replies short (2-3 sentences). "
        "Never mention databases or listings."
    )
    contexts = {
        "greeting": f"Greet warmly, welcome to {COMPANY_NAME}, offer to find businesses or guide on IBH.",
        "farewell": "Wish them well, invite them back.",
        "thanks":   "Accept graciously, offer further help.",
        "help":     f"Explain: 1) Find businesses 2) Guide on IBH platform 3) Help list/verify business.",
        "unknown":  "Ask warmly if they want to find a business or need help with IBH.",
    }
    prompt = f"""
Situation: {contexts.get(intent, contexts['unknown'])}
User Message: "{user_msg}"
Reply:
"""
    try:
        return ask_llama(prompt, system=system).strip()
    except Exception as e:
        logger.error("handle_conversational failed: %s", e)
        return f"Hello! Welcome to {COMPANY_NAME}! 😊 How can I help you today?"


# ─────────────────────────────────────────────
# GENERAL ADVICE HANDLER
# ─────────────────────────────────────────────
GENERAL_ADVICE = {
    "it": {
        "keywords": ["it", "software", "tech", "technology", "computer", "digital", "app", "website"],
        "tips": [
            "Build a strong online presence with a professional website and active social media.",
            "Focus on client retention through excellent after-sales support and follow-ups.",
            "Stay updated with the latest technologies, certifications, and industry trends.",
            "Collect and showcase customer reviews, case studies, and success stories.",
            "Network with other businesses, attend tech events, and explore partnerships.",
        ]
    },
    "restaurant": {
        "keywords": ["restaurant", "food", "cafe", "catering", "hotel", "dining", "eat"],
        "tips": [
            "Maintain consistent food quality and hygiene standards at all times.",
            "Use social media to showcase your dishes with attractive photos and videos.",
            "Offer loyalty programs, discounts, and special deals to retain customers.",
            "Collect customer feedback and act on it to improve your menu and service.",
            "Partner with food delivery platforms to reach a wider audience.",
        ]
    },
    "retail": {
        "keywords": ["shop", "store", "retail", "apparel", "clothing", "fashion", "sell"],
        "tips": [
            "Keep your inventory updated with trending products and seasonal items.",
            "Create an engaging storefront both physically and online.",
            "Run promotions, sales, and loyalty programs to attract repeat customers.",
            "Use social media marketing to showcase products and reach new buyers.",
            "Provide excellent customer service to build trust and word-of-mouth referrals.",
        ]
    },
    "marketing": {
        "keywords": ["marketing", "advertise", "promotion", "brand", "social media", "digital marketing"],
        "tips": [
            "Define your target audience clearly before planning any marketing campaign.",
            "Use a mix of social media, email, and content marketing for best results.",
            "Create valuable content that educates and engages your potential customers.",
            "Track your marketing metrics regularly and optimize based on what works.",
            "Invest in local SEO so customers in your city can find you easily online.",
        ]
    },
    "customer": {
        "keywords": ["customer", "client", "visitors", "attract", "more customers", "get customers"],
        "tips": [
            "Provide exceptional service that makes customers want to return and refer others.",
            "Ask satisfied customers for reviews and testimonials on Google and social media.",
            "Run referral programs that reward customers for bringing new clients.",
            "Stay active on social media and engage with your audience regularly.",
            "Offer first-time discounts or free trials to attract new customers.",
        ]
    },
    "default": {
        "keywords": [],
        "tips": [
            "Build a strong online presence with a professional website and social media profiles.",
            "Focus on delivering exceptional customer service to build loyalty and referrals.",
            "Invest in digital marketing — social media, email, and local SEO.",
            "Keep your skills, products, and services updated with market trends.",
            "Collect customer reviews and use feedback to continuously improve.",
        ]
    }
}


def handle_general(user_msg: str) -> str:
    msg     = user_msg.lower()
    matched = GENERAL_ADVICE["default"]
    for category, data in GENERAL_ADVICE.items():
        if category == "default":
            continue
        if any(kw in msg for kw in data["keywords"]):
            matched = data
            break

    tips     = matched["tips"]
    response = "Here are 5 great tips to help you:\n\n"
    for i, tip in enumerate(tips, 1):
        response += f"{i}. {tip}\n"
    response += "\nExplore related businesses in your city on IBH for more inspiration!"
    return response.strip()


# ─────────────────────────────────────────────
# MAIN CHAT
# ─────────────────────────────────────────────
def chat(user_msg: str) -> str:
    user_msg = user_msg.strip()
    if not user_msg:
        return "I didn't catch that. Could you please type your question?"

    intent = detect_intent(user_msg)
    print(f"[CHAT] Intent: {intent} | Message: {user_msg}")

    if intent == "guidance":
        return handle_guidance(user_msg)

    if intent in {"greeting", "farewell", "thanks", "help"}:
        return handle_conversational(intent, user_msg)

    if intent == "general":
        return handle_general(user_msg)

    if intent == "unknown":
        if len(user_msg.split()) <= 4:
            return handle_conversational("greeting", user_msg)
        return handle_general(user_msg)

    # ── Search pipeline ──
    db_context = ""

    # 1st — Semantic search
    db_context = semantic_search(user_msg)

    # 2nd — SQL
    if not db_context.strip():
        sql = generate_sql(user_msg)
        if sql:
            db_context = run_sql(sql)

    # 3rd — ORM fallback
    if not db_context.strip():
        db_context = get_dynamic_data(user_msg)

    # 4th — No results
    if not db_context.strip():
        if is_guidance_question(user_msg):
            return handle_guidance(user_msg)
        return (
            "I couldn't find any businesses matching your search. 😔\n"
            "Try a different category or city name.\n"
            "Example: 'restaurants in Surat' or 'IT services in Ahmedabad'"
        )

    final_prompt = f"""
You are a friendly assistant for Indian Business Hub (IBH).

Use ONLY the data below to answer.
- Number each result clearly.
- Include phone numbers.
- End with a helpful suggestion.

Data:
{db_context}

User Question: {user_msg}
Answer:
"""
    try:
        return ask_llama(final_prompt).strip()
    except Exception as e:
        logger.error("Final LLaMA call failed: %s", e)
        return "Something went wrong. Please try again."


def reload_embeddings():
    sync_businesses_to_pinecone()


def get_categories(page: int = 1, limit: int = 10) -> dict:
    start = (page - 1) * limit
    end   = start + limit
    total = Category.objects.count()
    names = list(Category.objects.values_list("name", flat=True)[start:end])
    return {"total": total, "page": page, "categories": names}