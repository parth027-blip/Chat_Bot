from django.db import models


# ─────────────────────────────────────────────
# CHAT LOG
# ─────────────────────────────────────────────
class ChatApp(models.Model):
    question = models.CharField(max_length=500)
    answer = models.TextField()
    is_answered = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "chatapp"
        ordering = ["-created_at"]

    def __str__(self):
        return self.question


# ─────────────────────────────────────────────
# CATEGORY
# ─────────────────────────────────────────────
class Category(models.Model):
    name = models.CharField(max_length=200)
    description = models.TextField(null=True, blank=True)
    keywords = models.TextField(null=True, blank=True)
    thumbnail = models.TextField(null=True, blank=True)
    is_active = models.IntegerField(default=1)
    is_featured = models.IntegerField(default=0)
    sort_order = models.IntegerField(default=0)
    category_type = models.CharField(
        max_length=20,
        choices=[
            ("festival", "Festival"),
            ("daily", "Daily"),
            ("greeting", "Greeting"),
            ("motivation", "Motivation"),
            ("business", "Business"),
        ],
        default="business",
    )
    created_at = models.DateTimeField(auto_now_add=True, null=True)
    updated_at = models.DateTimeField(null=True, blank=True)
    deleted_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = "categories"
        managed = False
        ordering = ["sort_order", "name"]

    def __str__(self):
        return self.name


# ─────────────────────────────────────────────
# CITY
# ─────────────────────────────────────────────
class City(models.Model):
    city = models.CharField(max_length=255)
    state_id = models.IntegerField(default=0)
    is_top = models.IntegerField(default=0)
    image = models.TextField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True, null=True)
    updated_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = "cities"
        managed = False
        ordering = ["city"]

    def __str__(self):
        return self.city


# ─────────────────────────────────────────────
# USER (Business Listing)
# ─────────────────────────────────────────────
class User(models.Model):
    name = models.CharField(max_length=199, null=True, blank=True)
    email = models.CharField(max_length=199)
    phone = models.CharField(max_length=199, null=True, blank=True)
    business_name = models.CharField(max_length=199, null=True, blank=True)
    address = models.TextField(null=True, blank=True)
    description = models.TextField(null=True, blank=True)
    website = models.TextField(null=True, blank=True)

    # Location
    city = models.ForeignKey(
        City,
        on_delete=models.CASCADE,
        db_column="city",
        null=True,
        blank=True,
    )
    state = models.IntegerField(null=True, blank=True)
    pincode = models.CharField(max_length=199, null=True, blank=True)
    latitude = models.DecimalField(max_digits=10, decimal_places=7, null=True, blank=True)
    longitude = models.DecimalField(max_digits=10, decimal_places=7, null=True, blank=True)

    # Category
    category = models.ForeignKey(
        Category,
        on_delete=models.CASCADE,
        db_column="category_id",
        null=True,
        blank=True,
    )

    # Social
    facebook = models.TextField(null=True, blank=True)
    linkedin = models.TextField(null=True, blank=True)
    whatsapp_no = models.CharField(max_length=199, null=True, blank=True)
    visiting_card_url = models.TextField(null=True, blank=True)

    # Status flags
    is_active = models.IntegerField(default=1)
    is_verified = models.IntegerField(default=0)
    is_email_verified = models.IntegerField(default=0)
    is_profile_complete = models.IntegerField(default=0)
    is_document_uploaded = models.IntegerField(default=0)
    is_my_business_uploaded = models.IntegerField(default=0)
    profile_percentage = models.IntegerField(default=0)

    # Meta
    added_from = models.CharField(max_length=10, default="app")
    user_from = models.CharField(max_length=10, default="app")
    login_type = models.CharField(max_length=10, default="email")
    created_at = models.DateTimeField(auto_now_add=True, null=True)
    updated_at = models.DateTimeField(null=True, blank=True)
    deleted_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = "users"
        managed = False
        ordering = ["-created_at"]

    def __str__(self):
        return self.business_name or self.name or self.email

    @property
    def is_deleted(self):
        return self.deleted_at is not None

# ─────────────────────────────────────────────
# SERVICE
# ─────────────────────────────────────────────
class Service(models.Model):
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        db_column="user_id",
        related_name="services",
    )
    category = models.ForeignKey(
        Category,
        on_delete=models.CASCADE,
        db_column="category_id",
        null=True,
        blank=True,
    )
    service_title = models.CharField(max_length=199)
    description = models.TextField(null=True, blank=True)
    keywords = models.TextField(null=True, blank=True)
    thumbnail = models.TextField(null=True, blank=True)
    is_active = models.IntegerField(default=1)
    created_at = models.DateTimeField(auto_now_add=True, null=True)
    updated_at = models.DateTimeField(null=True, blank=True)
    deleted_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = "services"
        managed = False
        ordering = ["-created_at"]

    def __str__(self):
        return self.service_title

    @property
    def is_deleted(self):
        return self.deleted_at is not None


# ─────────────────────────────────────────────
# PROFILE DAILY VISITS
# ─────────────────────────────────────────────
class ProfileDailyVisit(models.Model):
    profile = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        db_column="profile_id",
        related_name="daily_visits",
    )
    visit_date = models.DateField()
    visits = models.IntegerField(default=0)

    class Meta:
        db_table = "profile_daily_visits"
        managed = False
        ordering = ["-visit_date"]
        # One record per profile per day
        unique_together = ("profile", "visit_date")

    def __str__(self):
        return f"{self.profile_id} — {self.visit_date} — {self.visits} visits"
    
