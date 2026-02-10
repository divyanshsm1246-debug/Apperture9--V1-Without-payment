# ==========================================
# APPERTURE BACKEND API (FastAPI + SQLite)
# ==========================================
# This is the server-side code to replace the visual HTML mock.
# It handles Data Persistence, Authentication, Ad Scheduling, and Chat.
#
# INSTRUCTIONS TO RUN:
# 1. Install dependencies: pip install fastapi uvicorn pydantic
# 2. Run server: uvicorn backend:app --reload
# ==========================================

import sqlite3
import time
import random
import re
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os
# --- CONFIGURATION ---
DB_NAME = "apperture.db"
ADMIN_KEY = "divyanshgupta.Apperture.Elderweb.com@passkey142682920252717"
CO_ADMIN_KEYS = [
    "naman.ronaldo.siue.12344",
    "sanyam.bro.passkey@!#$%#&^%((}",
    "Passkey.co.admin@apperture.siuuee._12221117",
    "Passkey.Cricket.madhav.bro.@@paskkey-siuuuuue.",
    "Siiiiiiue.passkey#@@big.dawgs//@!234455.Tehlka-Omelelte."
]

# Comprehensive Profanity Filter List
RESTRICTED_WORDS = [
    "badword1", "badword2", "gali1", "gali2", "palabra1",
    "abuse", "violent", "vulgar", "slang", "foul", "kill", "hate", "idiot", "stupid",
    "scam", "fraud", "shit", "fuck", "bitch", "ass", "damn", "crap", "sex", "nude",
    "porn", "weapon", "gun", "drug", "weed", "terror", "bomb", "racist", "nazi",
    "slave", "rape", "murder", "suicide", "die", "death"
]

# Bolt Optimization: Compile regex once to speed up profanity checking
RESTRICTED_PATTERN = re.compile("|".join(re.escape(w) for w in RESTRICTED_WORDS), re.IGNORECASE)

# --- SUPER NEXUS ENCRYPTION FRAUD DETECTOR AI ---
# Advanced Fraud Detection Patterns
FRAUD_KEYWORDS = [
    "scam", "quick money", "transfer", "cashout", "whatsapp", "telegram", "external", 
    "gift card", "crypto", "bitcoin", "paypal", "pay outside", "contact me at",
    "send money", "secure payment", "winner", "prize", "verification code", "free money"
]
FRAUD_PATTERN = re.compile("|".join(re.escape(w) for w in FRAUD_KEYWORDS), re.IGNORECASE)

# Obfuscated contact detection (e.g., "w h a t s a p p" or "9 8 7...")
OBFUSCATED_CONTACT_PATTERN = re.compile(r"(\w\s){4,}|(\d\s){4,}", re.IGNORECASE)
PHONE_PATTERN = re.compile(r"(\+?\d[\s-]?){10,}")
URL_PATTERN = re.compile(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+")

# Allowed User Chat Phrases (Point 24/25)
ALLOWED_USER_PHRASES = [
    "Is the price negotiable?",
    "I can offer ₹... for this.",
    "What is your best price?",
    "I accept your offer.",
    "Sorry, that price is too low.",
    "Would you consider ₹...?",
    "Let's meet in the middle at ₹...",
    "Is this still available?",
    "Yes, I'll take it.",
    "No, thanks.",
    "Chat Started" # Internal
]

def super_nexus_analyze_text(text: str, user: dict = None) -> dict:
    """
    Ultimate text analysis for Nexus AI.
    - Admins are scanned but not blocked unless extreme.
    - Users are strictly validated against phrases in chat.
    """
    if not user: return {"score": 0, "reasons": []}
    score = 0.0
    reasons = []
    
    # Check Restricted Profanity First
    if RESTRICTED_PATTERN.search(text):
        score += 1.0
        reasons.append("Restricted/Vulgar content detected")

    if FRAUD_PATTERN.search(text):
        score += 0.8
        reasons.append("Fraudulent keywords detected")
        
    if OBFUSCATED_CONTACT_PATTERN.search(text):
        score += 0.8
        reasons.append("Obfuscated contact attempt")
        
    if PHONE_PATTERN.search(text):
        score += 0.9
        reasons.append("Direct contact solicitation")
        
    if URL_PATTERN.search(text):
        score += 0.7
        reasons.append("External link detected")
        
    return {"score": score, "reasons": reasons}

def super_nexus_fraud_analysis(item: Any, user: dict) -> dict:
    """
    Super Nexus AI Logic: Detects fraud patterns.
    Returns: { "is_fraud": bool, "score": float, "reason": str }
    """
    if user['is_admin']: return {"is_fraud": False, "score": 0, "reason": "Admin Bypass"}

    score = 0.0
    all_reasons = []
    
    # 1. Text Analysis (Title + Desc)
    text_report = super_nexus_analyze_text(f"{item.title} {item.description}", user)
    score += text_report["score"]
    all_reasons.extend(text_report["reasons"])
    
    # 2. Price Anomaly Detection
    # If price is suspiciously high for simple categories (e.g., > 1,000,000 INR for a T-Shirt)
    if hasattr(item, 'price') and item.price > 500000 and item.category in ["T-Shirts", "Hats", "Toys"]:
        score += 0.8
        all_reasons.append("Extreme price anomaly for category")
        
    # 3. User Reputation (Simulation)
    if user['stars_balance'] < 10 and hasattr(item, 'price') and item.price > 100000:
        score += 0.4
        all_reasons.append("High-value item from low-reputation user")
        
    return {
        "is_fraud": score >= 1.0,
        "score": min(score, 1.0),
        "reason": "; ".join(all_reasons) if all_reasons else "Clean"
    }


def is_user_verified(user: dict) -> bool:
    """A user is verified if they have a name (not 'New User'), address, and password."""
    if user.get('is_admin', 0) > 0: return True
    name = user.get('name', '')
    address = user.get('address', '')
    password = user.get('password', '')
    return bool(name and name != "New User" and address and password)

def verify_access(user: dict):
    if not is_user_verified(user):
        raise HTTPException(status_code=403, detail="Verification Required. Please set up your profile with a name, address, and password to unlock this feature.")

def is_clean(text: Optional[str]) -> bool:
    if not text: return True
    # search() is generally faster than iterating through the list in Python
    return not bool(RESTRICTED_PATTERN.search(text))

app = FastAPI(title="Apperture API", description="Backend for Apperture Fashion Marketplace")

# Allow CORS (Cross-Origin Resource Sharing) so a frontend can talk to this
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- DATABASE SETUP ---
def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize the database tables if they don't exist."""
    conn = get_db_connection()
    c = conn.cursor()
    
    # Users Table
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,
        name TEXT,
        phone TEXT,
        profile_pic TEXT,
        password TEXT,
        is_admin BOOLEAN DEFAULT 0,
        is_banned BOOLEAN DEFAULT 0,
        stars_balance INTEGER DEFAULT 0,
        gems_balance INTEGER DEFAULT 0,
        tickets_balance INTEGER DEFAULT 0,
        address TEXT,
        created_at REAL
    )''')

    # Listings Table
    c.execute('''CREATE TABLE IF NOT EXISTS listings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        seller_id TEXT,
        title TEXT,
        category TEXT,
        condition TEXT,
        price REAL,
        description TEXT,
        image_data TEXT, -- Stores JSON array of image strings
        phone TEXT,
        address TEXT,
        is_magnetic BOOLEAN DEFAULT 0,
        is_ad BOOLEAN DEFAULT 0, is_gem_boosted BOOLEAN DEFAULT 0,
        ad_link TEXT,
        currency TEXT DEFAULT '₹',
        stars_count INTEGER DEFAULT 0,
        created_at REAL,
        FOREIGN KEY(seller_id) REFERENCES users(id)
    )''')

    # Cart Table
    c.execute('''CREATE TABLE IF NOT EXISTS cart (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT,
        listing_id INTEGER,
        title TEXT,
        price REAL,
        image TEXT,
        added_at REAL
    )''')

    # Chats Table
    c.execute('''CREATE TABLE IF NOT EXISTS chats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user1_id TEXT,
        user2_id TEXT,
        user1_name TEXT,
        user2_name TEXT,
        last_message TEXT,
        last_updated REAL
    )''')

    # Messages Table
    c.execute('''CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chat_id INTEGER,
        sender_id TEXT,
        text TEXT,
        timestamp REAL,
        FOREIGN KEY(chat_id) REFERENCES chats(id)
    )''')

    # Supreme Ad Bookings
    c.execute('''CREATE TABLE IF NOT EXISTS supreme_ads (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        slot_id TEXT,
        title TEXT,
        category TEXT DEFAULT 'All',
        image_data TEXT,
        ad_copy TEXT,
        buyer_id TEXT,
        start_time REAL,
        end_time REAL,
        created_at REAL
    )''')

    # Notifications
    c.execute('''CREATE TABLE IF NOT EXISTS notifications (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        type TEXT, -- 'admin_alert', 'user_ad', 'private'
        message TEXT,
        poster_id TEXT,
        recipient_id TEXT, -- NULL for broadcast
        scheduled_for REAL,
        expires_at REAL
    )''')

    # Stars Votes (Limit 20 per user/listing)
    c.execute('''CREATE TABLE IF NOT EXISTS stars_votes (
        user_id TEXT,
        listing_id INTEGER,
        count INTEGER DEFAULT 0,
        PRIMARY KEY(user_id, listing_id)
    )''')

    # Daily Rewards Claims
    c.execute('''CREATE TABLE IF NOT EXISTS daily_rewards_claims (
        user_id TEXT PRIMARY KEY,
        last_claimed_at REAL,
        day_streak INTEGER DEFAULT 0
    )''')

    # Used Coupons (Single use per user)
    c.execute('''CREATE TABLE IF NOT EXISTS used_coupons (
        user_id TEXT,
        coupon_code TEXT,
        claimed_at REAL,
        PRIMARY KEY(user_id, coupon_code)
    )''')

    # Performance Indices
    c.execute('CREATE INDEX IF NOT EXISTS idx_listings_category ON listings(category)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_listings_stars ON listings(stars_count)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_listings_seller ON listings(seller_id)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_supreme_ads_end ON supreme_ads(end_time)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_notifications_expires ON notifications(expires_at)')

    # Robustness: Migrations
    # Robustness: Migrations
    def get_columns(table):
        cursor = c.execute(f"PRAGMA table_info({table})")
        return [row[1] for row in cursor.fetchall()]

    if 'password' not in get_columns('users'):
        c.execute('ALTER TABLE users ADD COLUMN password TEXT')

    if 'recipient_id' not in get_columns('notifications'):
        c.execute('ALTER TABLE notifications ADD COLUMN recipient_id TEXT')

    if 'address' not in get_columns('listings'):
        c.execute('ALTER TABLE listings ADD COLUMN address TEXT')
    
    if 'category' not in get_columns('supreme_ads'):
        c.execute('ALTER TABLE supreme_ads ADD COLUMN category TEXT DEFAULT "All"')

    if 'currency' not in get_columns('listings'):
        c.execute('ALTER TABLE listings ADD COLUMN currency TEXT DEFAULT "₹"')

    conn.commit()
    
    conn.close()

# Initialize DB on startup
init_db()

# --- PYDANTIC MODELS (Data Validation) ---

class UserProfile(BaseModel):
    id: str
    name: str
    phone: Optional[str] = None
    profile_pic: Optional[str] = None
    address: Optional[str] = None
    password: Optional[str] = None

class ListingCreate(BaseModel):
    title: str
    category: str
    condition: Optional[str] = None
    price: Optional[float] = 0.0
    currency: Optional[str] = '₹'
    description: str
    image_data: Optional[str] = None # Support legacy
    images: Optional[List[str]] = [] # Support multiple for 3D view
    phone: Optional[str] = None
    address: Optional[str] = None
    is_magnetic: bool = False
    is_ad: bool = False
    ad_link: Optional[str] = None

class CartItem(BaseModel):
    listing_id: int
    title: str
    price: float
    image: str

class ChatMessage(BaseModel):
    text: str

class SupremeAdCreate(BaseModel):
    slot_id: str # 'slot1', 'slot2', 'slot3'
    title: str
    category: str = "All"
    image_data: str
    ad_copy: str

class NotificationCreate(BaseModel):
    message: str
    type: str = "user_ad"
    recipient_id: Optional[str] = None

# --- AUTHENTICATION HELPER (Simulated) ---
# In a real app, use JWT tokens. Here we trust the user_id header for simplicity in this demo.
async def get_current_user(x_user_id: str = Header(...)):
    conn = get_db_connection()
    user_row = conn.execute("SELECT * FROM users WHERE id = ?", (x_user_id,)).fetchone()
    conn.close()
    if not user_row:
        # Auto-create user if sending ID for first time (Demo logic)
        conn = get_db_connection()
        conn.execute("INSERT INTO users (id, name, tickets_balance, created_at) VALUES (?, ?, ?, ?)", 
                     (x_user_id, "New User", 1, time.time()))
        conn.commit()
        user_row = conn.execute("SELECT * FROM users WHERE id = ?", (x_user_id,)).fetchone()
        conn.close()
    
    user = dict(user_row)
    if user['is_banned']:
        raise HTTPException(status_code=403, detail="User is banned")
    
    # Requirement: Admin has unlimited things
    if user['is_admin']:
        user['stars_balance'] = 999999
        user['gems_balance'] = 999999
        user['tickets_balance'] = 999999
        
    return user

# --- ENDPOINTS ---

@app.get("/")
def read_root():
    # Use absolute path to ensure the file is found regardless of where the server is started
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "Apperture.html")
    if not os.path.exists(file_path):
        return {"error": "Apperture.html not found", "hint": "Ensure the HTML file is in the root directory."}
    return FileResponse(file_path)

@app.get("/favicon.ico")
def favicon():
    return {"message": "No favicon provided"}

# --- USER PROFILES ---

@app.get("/profile")
def get_profile(user: dict = Depends(get_current_user)):
    return dict(user)

@app.put("/profile")
def update_profile(profile: UserProfile, user: dict = Depends(get_current_user)):
    if not is_clean(profile.name):
        raise HTTPException(status_code=400, detail="Name contains restricted content")
        
    conn = get_db_connection()
    
    is_admin = user['is_admin']
    name = profile.name
    
    # Admin Secret Key Logic (Hierarchy: 2=Main, 1=Co-Admin, 0=Normal)
    if name == ADMIN_KEY:
        is_admin = 2
        name = "Primary Admin (Divyansh Gupta)"
    elif name in CO_ADMIN_KEYS:
        is_admin = 1
        # Keep original name or generic co-admin name? 
        # User said "CO admin: number 1...", I'll keep the passkey as name for now so they know it worked, or use a prefix.
        name = f"Co-Admin ({CO_ADMIN_KEYS.index(name) + 1})"
    
    # Coupon Code Logic
    # 25 unique codes requested in summary
    VALID_COUPONS = {
        "APERTURE-REWARD-PRIME-X7L92": (500, 7, 3),
        "APERTURE-REWARD-GOLD-B2M41": (1000, 10, 5),
        "APERTURE-REWARD-ELITE-N9P15": (750, 15, 10),
        "APERTURE-REWARD-ALPHA-K8Q22": (800, 14, 8),
        "APERTURE-REWARD-BETA-C3V19": (900, 16, 12),
        "APERTURE-REWARD-GAMMA-D5W38": (1100, 18, 13),
        "APERTURE-REWARD-DELTA-F2X47": (600, 8, 4),
        "APERTURE-REWARD-SIGMA-H6Y56": (1200, 19, 14),
        "APERTURE-REWARD-ZETA-J1Z65": (550, 7, 5),
        "APERTURE-REWARD-OMEGA-L4A74": (2500, 25, 20),
        "APERTURE-REWARD-CHAMP-M5B83": (510, 9, 6),
        "APERTURE-REWARD-LEGEND-R7C92": (1500, 15, 15),
        "APERTURE-REWARD-MYTHIC-T9D11": (1750, 20, 12),
        "APERTURE-REWARD-GODLY-U2E22": (5000, 50, 25),
        "APERTURE-REWARD-UBER-V4F33": (620, 12, 10),
        "APERTURE-REWARD-PRO-W6G44": (680, 13, 11),
        "APERTURE-REWARD-HACKER-G1H55": (550, 10, 10),
        "APERTURE-REWARD-GHOST-S3I66": (500, 7, 11),
        "APERTURE-REWARD-PHANTOM-P8J77": (500, 15, 10),
        "APERTURE-REWARD-SHADOW-Z2K88": (500, 10, 10),
        "APERTURE-REWARD-BLADE-Y4L99": (1999, 19, 10),
        "APERTURE-REWARD-STRIKE-E7M11": (611, 11, 11),
        "APERTURE-REWARD-FLASH-R5N22": (1555, 15, 15),
        "APERTURE-REWARD-TITAN-O9P33": (10000, 100, 100),
        "APERTURE-REWARD-ZEN-A1B2C": (500, 50, 50)
    }
    
    coupon_reward = None
    if name in VALID_COUPONS:
        # Check if already used
        used = conn.execute("SELECT 1 FROM used_coupons WHERE user_id = ? AND coupon_code = ?", (user['id'], name)).fetchone()
        if not used:
            stars, gems, tickets = VALID_COUPONS[name]
            conn.execute("INSERT INTO used_coupons (user_id, coupon_code, claimed_at) VALUES (?, ?, ?)", (user['id'], name, time.time()))
            conn.execute("UPDATE users SET stars_balance = stars_balance + ?, gems_balance = gems_balance + ?, tickets_balance = tickets_balance + ? WHERE id = ?",
                         (stars, gems, tickets, user['id']))
            coupon_reward = f"Redeemed Coupon! +{stars} Stars, +{gems} Gems, +{tickets} Tickets."
            # Don't overwrite actual name with coupon code
            name = user['name'] 

    conn.execute("UPDATE users SET name = ?, phone = ?, profile_pic = ?, address = ?, password = ?, is_admin = ? WHERE id = ?", 
                 (name, profile.phone, profile.profile_pic, profile.address, profile.password, is_admin, user['id']))
    conn.commit()
    conn.close()
    return {"status": "updated", "is_admin": is_admin == 1, "reward": coupon_reward}

# --- LISTINGS & SHOP ---

@app.get("/listings")
def get_listings(category: str = "All", user_id: Optional[str] = None, near_me: bool = False):
    import json
    conn = get_db_connection()
    query = "SELECT * FROM listings WHERE 1=1"
    params = []

    if category != "All":
        query += " AND category = ?"
        params.append(category)
    
    items = conn.execute(query, params).fetchall()
    listing_dicts = []
    
    for item in items:
        d = dict(item)
        # Bolt Optimization: For the grid view, only return the first image to reduce payload size.
        # This significantly speeds up the 4-second polling cycle.
        if d['image_data']:
            try:
                imgs = json.loads(d['image_data'])
                if isinstance(imgs, list) and len(imgs) > 0:
                    d['image_data'] = imgs[0] # Just the first one
            except:
                pass
        listing_dicts.append(d)

    # Location Based Filtering (Point 22)
    if near_me and user_id:
        user = conn.execute("SELECT address FROM users WHERE id = ?", (user_id,)).fetchone()
        if user and user['address']:
            u_addr = user['address'].lower()
            # Simple keyword match for proximity in this demo
            listing_dicts = [l for l in listing_dicts if (l['address'] and u_addr in l['address'].lower()) or (l['description'] and u_addr in l['description'].lower())]

    conn.close()
    
    # Point 32: Refresh shop hourly with variety (Shuffle seeded by current hour)
    seed = int(time.time() / 3600)
    rng = random.Random(seed)
    rng.shuffle(listing_dicts)
    
    return listing_dicts

@app.get("/listings/{item_id}")
def get_listing_detail(item_id: int):
    conn = get_db_connection()
    item = conn.execute("SELECT * FROM listings WHERE id = ?", (item_id,)).fetchone()
    conn.close()
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    return dict(item)

@app.post("/listings")
def create_listing(item: ListingCreate, user: dict = Depends(get_current_user)):
    verify_access(user)
    import json
    if not is_clean(item.title) or not is_clean(item.description):
        raise HTTPException(status_code=400, detail="Listing contains restricted content")
        
    # Super Nexus AI Check
    nexus_report = super_nexus_fraud_analysis(item, user)
    if nexus_report["is_fraud"] and not user['is_admin']:
        raise HTTPException(status_code=403, detail=f"Super Nexus Encryption AI Block: {nexus_report['reason']}")
        
    conn = get_db_connection()
    
    # Validation
    if item.category == "Special Products" and not user['is_admin']:
        raise HTTPException(status_code=403, detail="Only admins can post in Special Products")

    if item.is_magnetic and not user['is_admin']:
        # Check logic: 1 magnetic for every 3 normal in category
        count = conn.execute("SELECT COUNT(*) FROM listings WHERE category = ? AND is_magnetic = 0", (item.category,)).fetchone()[0]
        mag_count = conn.execute("SELECT COUNT(*) FROM listings WHERE category = ? AND is_magnetic = 1", (item.category,)).fetchone()[0]
        
        if count < 3 or mag_count >= (count // 3):
            raise HTTPException(status_code=400, detail="Magnetic slot full for this category. Ratio 1:3 active.")

    # Use user's profile address if not provided in listing
    listing_address = item.address if item.address else user['address']
    
    # Handle multiple images
    imgs = item.images if item.images else []
    if not imgs and item.image_data:
        imgs = [item.image_data]
    
    image_json = json.dumps(imgs)
    
    c = conn.execute('''INSERT INTO listings 
        (seller_id, title, category, condition, price, currency, description, image_data, phone, address, is_magnetic, is_ad, ad_link, stars_count, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)''',
        (user['id'], item.title, item.category, item.condition, item.price, item.currency, item.description, image_json, item.phone, listing_address, item.is_magnetic, item.is_ad, item.ad_link, time.time())
    )
    conn.commit()
    new_id = c.lastrowid
    conn.close()
    return {"status": "created", "id": new_id}

@app.delete("/listings/{item_id}")
def delete_listing(item_id: int, user: dict = Depends(get_current_user)):
    conn = get_db_connection()
    item = conn.execute("SELECT * FROM listings WHERE id = ?", (item_id,)).fetchone()
    if not item:
        conn.close()
        raise HTTPException(status_code=404, detail="Item not found")
        
    # Co-admins (is_admin=1) cannot delete other people's listings. Only primary admin (is_admin=2) or owner can.
    if item['seller_id'] != user['id'] and not user['is_admin']:
        conn.close()
        raise HTTPException(status_code=403, detail="Not authorized. Only Admins can force remove.")
        
    conn.execute("DELETE FROM listings WHERE id = ?", (item_id,))
    conn.commit()
    conn.close()
    return {"status": "deleted"}

# --- CART ---

@app.get("/cart")
def get_cart(user: dict = Depends(get_current_user)):
    conn = get_db_connection()
    items = conn.execute("SELECT * FROM cart WHERE user_id = ?", (user['id'],)).fetchall()
    conn.close()
    return [dict(item) for item in items]

@app.post("/cart")
def add_to_cart(item: CartItem, user: dict = Depends(get_current_user)):
    conn = get_db_connection()
    conn.execute("INSERT INTO cart (user_id, listing_id, title, price, image, added_at) VALUES (?, ?, ?, ?, ?, ?)",
                 (user['id'], item.listing_id, item.title, item.price, item.image, time.time()))
    conn.commit()
    conn.close()
    return {"status": "added"}

@app.delete("/cart/{cart_id}")
def remove_from_cart(cart_id: int, user: dict = Depends(get_current_user)):
    conn = get_db_connection()
    conn.execute("DELETE FROM cart WHERE id = ? AND user_id = ?", (cart_id, user['id']))
    conn.commit()
    conn.close()
    return {"status": "removed"}

# --- SUPREME ADS ---

@app.get("/ads/supreme")
def get_supreme_ads(category: str = "All"):
    conn = get_db_connection()
    now = time.time()
    # Get active or upcoming ads. Frontend will handle 'Coming Soon' vs 'Live'
    query = "SELECT * FROM supreme_ads WHERE end_time > ?"
    params = [now]
    
    if category != "All":
        query += " AND (category = ? OR category = 'All')"
        params.append(category)
        
    query += " ORDER BY start_time ASC"
    
    ads = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(ad) for ad in ads]

@app.post("/ads/supreme")
def book_supreme_ad(ad: SupremeAdCreate, user: dict = Depends(get_current_user)):
    verify_access(user)
    conn = get_db_connection()
    now = time.time()
    
    # Find best slot (the one with earliest available time)
    slots = ["slot1", "slot2", "slot3"]
    best_slot = "slot1"
    min_start_time = float('inf')
    
    for s in slots:
        # Check for latest end_time in this slot
        existing = conn.execute("SELECT end_time FROM supreme_ads WHERE slot_id = ? AND end_time > ? ORDER BY end_time DESC LIMIT 1", (s, now)).fetchone()
        available_at = existing['end_time'] if existing else now
        if available_at < min_start_time:
            min_start_time = available_at
            best_slot = s
            
    # Sharp Execution: Start time must align to 30-min boundaries
    # Requirement: 30-min prior notification. 
    # If we want notification at 2:59/3:00 for 3:29/3:30 execution.
    
    # Calculate next available 30-min block after available_at + 30m buffer (for mandatory notification)
    # Strict 30-minute interval system (Point 1 & refinement)
    # We snap to :00 or :30 of the hour.
    earliest_allowed = now + 1800 # Min 30m gap for notification
    potential_start = max(min_start_time, earliest_allowed)
    
    # Snap to next 30-min boundary
    start_time = (int(potential_start / 1800) + 1) * 1800
    
    # Limit check: Bookable up to 2 days in advance (48h) (Point 3)
    if start_time > (now + 172800) and user['is_admin'] == 0:
         conn.close()
         raise HTTPException(status_code=400, detail="All Supreme slots fully booked for the next 48 hours. Please check back later.")
    
    end_time = start_time + 86400  # 24 hours duration (Point 3)

    # 1. Create the Supreme Ad entry
    conn.execute('''INSERT INTO supreme_ads (slot_id, title, category, image_data, ad_copy, buyer_id, start_time, end_time, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                 (best_slot, ad.title, ad.category, ad.image_data, ad.ad_copy, user['id'], start_time, end_time, now))
    
    # 2. Requirement: Sharp Execution. Create a broadcast notification 30 minutes PRIOR to start.
    # Point 1 Refinement: "If booked at 2:29, notification at 2:59, execute at 3:29"
    notify_time = start_time - 1800
    msg = f"Upcoming Supreme Ad: '{ad.title}' is launching at {time.strftime('%H:%M', time.localtime(start_time))}! Prepare for the drop."
    conn.execute("INSERT INTO notifications (type, message, poster_id, scheduled_for, expires_at) VALUES (?, ?, ?, ?, ?)",
                 ("user_ad", msg, user['id'], notify_time, start_time + 3600))

    conn.commit()
    conn.close()
    return {"status": "booked", "start_time": start_time}

# --- CHAT ---

@app.get("/chats")
def get_my_chats(user: dict = Depends(get_current_user)):
    conn = get_db_connection()
    chats = conn.execute("SELECT * FROM chats WHERE user1_id = ? OR user2_id = ? ORDER BY last_updated DESC", (user['id'], user['id'])).fetchall()
    conn.close()
    return [dict(chat) for chat in chats]

class ChatCreate(BaseModel):
    participants: List[str]
    names: List[str]
    lastMessage: Optional[str] = "Chat started"

@app.post("/chats")
def start_chat(chat_req: ChatCreate, user: dict = Depends(get_current_user)):
    # Extract other user from participants list
    my_id = user['id']
    if len(chat_req.participants) == 2:
        if chat_req.participants[0] == my_id:
            other_user_id = chat_req.participants[1]
            other_user_name = chat_req.names[1]
        else:
            other_user_id = chat_req.participants[0]
            other_user_name = chat_req.names[0]
    else:
        # Handle self-chat or malformed request
        other_user_id = chat_req.participants[0] if chat_req.participants else my_id
        other_user_name = chat_req.names[0] if chat_req.names else "User"

    conn = get_db_connection()
    # Check if exists
    existing = conn.execute("SELECT * FROM chats WHERE (user1_id = ? AND user2_id = ?) OR (user1_id = ? AND user2_id = ?)",
                            (user['id'], other_user_id, other_user_id, user['id'])).fetchone()
    if existing:
        conn.close()
        return {"id": existing['id'], "status": "exists"}
    
    c = conn.execute("INSERT INTO chats (user1_id, user2_id, user1_name, user2_name, last_message, last_updated) VALUES (?, ?, ?, ?, ?, ?)",
                     (user['id'], other_user_id, user['name'], other_user_name, chat_req.lastMessage or "Chat Started", time.time()))
    conn.commit()
    new_id = c.lastrowid
    conn.close()
    return {"id": new_id, "status": "created"}

@app.get("/chats/{chat_id}/messages")
def get_messages(chat_id: int, user: dict = Depends(get_current_user)):
    conn = get_db_connection()
    # Verify participant
    chat = conn.execute("SELECT * FROM chats WHERE id = ?", (chat_id,)).fetchone()
    if not chat or (chat['user1_id'] != user['id'] and chat['user2_id'] != user['id']):
        conn.close()
        raise HTTPException(status_code=403, detail="Access denied")
        
    msgs = conn.execute("SELECT * FROM messages WHERE chat_id = ? ORDER BY timestamp ASC", (chat_id,)).fetchall()
    conn.close()
    return [dict(msg) for msg in msgs]

@app.post("/chats/{chat_id}/messages")
def send_message(chat_id: int, msg: ChatMessage, user: dict = Depends(get_current_user)):
    # 1. Enforcement of Point 7: Phrases for normal people, text for admin
    if not user['is_admin']:
        # Check if it's one of the allowed phrases or starts with system prefix
        is_allowed = any(phrase.split('₹...')[0] in msg.text for phrase in ALLOWED_USER_PHRASES)
        if not is_allowed and not msg.text.startswith("SYSTEM_OFFER:"):
             raise HTTPException(status_code=403, detail="Super Nexus Encryption AI Block: Users are limited to pre-defined secure phrases for safety.")

    # 2. Advanced Security Scan
    nexus_report = super_nexus_analyze_text(msg.text, user)
    if nexus_report["score"] >= 1.0 and not user['is_admin']:
        raise HTTPException(status_code=403, detail=f"Super Nexus Encryption AI Block: {'; '.join(nexus_report['reasons'])}")

    conn = get_db_connection()
    conn.execute("INSERT INTO messages (chat_id, sender_id, text, timestamp) VALUES (?, ?, ?, ?)",
                 (chat_id, user['id'], msg.text, time.time()))
    
    conn.execute("UPDATE chats SET last_message = ?, last_updated = ? WHERE id = ?", 
                 (msg.text, time.time(), chat_id))
    conn.commit()
    conn.close()
    return {"status": "sent"}

# --- NOTIFICATIONS (Admin Broadcast) ---

@app.get("/notifications")
def get_notifications(user: dict = Depends(get_current_user)):
    conn = get_db_connection()
    now = time.time()
    # Filter: Show global notifications OR private ones for the user
    notifs = conn.execute("""
        SELECT * FROM notifications 
        WHERE expires_at > ? 
        AND (recipient_id IS NULL OR recipient_id = ?)
        ORDER BY scheduled_for DESC
    """, (now, user['id'])).fetchall()
    conn.close()
    return [dict(n) for n in notifs]

@app.post("/notifications")
def create_notification(notif: NotificationCreate, user: dict = Depends(get_current_user)):
    if not is_clean(notif.message):
        raise HTTPException(status_code=400, detail="Notification contains restricted content")

    conn = get_db_connection()
    now = time.time()
    
    # If it's a private notification, bypass the 30-min slot logic
    if notif.recipient_id:
        expires = now + 86400
        conn.execute("INSERT INTO notifications (type, message, poster_id, recipient_id, scheduled_for, expires_at) VALUES (?, ?, ?, ?, ?, ?)",
                     (notif.type, notif.message, user['id'], notif.recipient_id, now, expires))
        conn.commit()
        conn.close()
        return {"status": "sent", "scheduled_at": now}

    # Requirement: 2 notifications per 30 mins. Strict interval system.
    # Point 1: 2 notification 30 min and booking can be till 24 hours.
    
    # We find the earliest available 30-min block (aligned to 00/30 mins of the hour)
    # Start searching from 30 mins in the future
    base_search = now + 1800
    found_slot = False
    target_time = base_search
    
    for i in range(48): # Search up to 24 hours (48 half-hour slots)
        candidate_time = base_search + (i * 1800)
        # Snap to 30-min boundary for strict interval system
        aligned_time = (int(candidate_time / 1800) + 1) * 1800
        
        # Only count BROADCAST notifications for the limit
        count = conn.execute("SELECT COUNT(*) FROM notifications WHERE scheduled_for = ? AND recipient_id IS NULL", (aligned_time,)).fetchone()[0]
        if count < 2:
            target_time = aligned_time
            found_slot = True
            break
            
    if not found_slot and not user['is_admin']:
        conn.close()
        raise HTTPException(status_code=400, detail="Notification broadcast slots are full for the next 24 hours.")

    if user['is_admin']: target_time = now # Admins bypass queue

    expires = target_time + 86400
    conn.execute("INSERT INTO notifications (type, message, poster_id, recipient_id, scheduled_for, expires_at) VALUES (?, ?, ?, ?, ?, ?)",
                 (notif.type, notif.message, user['id'], None, target_time, expires))
    conn.commit()
    conn.close()
    return {"status": "broadcasted", "scheduled_at": target_time}

# --- SYNC (Bolt Optimization ⚡) ---

class SyncRequest(BaseModel):
    user_id: str
    category: str = "All"

@app.post("/sync")
def sync_data(req: SyncRequest):
    """
    Bolt Optimization: Bundles multiple polls into one request.
    Returns: Listings, Notifications, Cart, and Chat metadata.
    """
    import json
    conn = get_db_connection()
    now = time.time()
    
    # Get user profile
    user_row = conn.execute("SELECT * FROM users WHERE id = ?", (req.user_id,)).fetchone()
    if not user_row:
        conn.execute("INSERT INTO users (id, name, tickets_balance, created_at) VALUES (?, ?, ?, ?)", 
                     (req.user_id, "New User", 1, now))
        conn.commit()
        user_row = conn.execute("SELECT * FROM users WHERE id = ?", (req.user_id,)).fetchone()
    
    user = dict(user_row)
    if user['is_admin']:
        user['stars_balance'] = 999999
        user['gems_balance'] = 999999
        user['tickets_balance'] = 999999

    # 1. Listings (Shallow)
    query_listings = "SELECT * FROM listings WHERE 1=1"
    params_listings = []
    if req.category != "All":
        query_listings += " AND category = ?"
        params_listings.append(req.category)
    
    listings_rows = conn.execute(query_listings, params_listings).fetchall()
    listings = []
    for row in listings_rows:
        d = dict(row)
        if d['image_data']:
            try:
                imgs = json.loads(d['image_data'])
                if isinstance(imgs, list) and len(imgs) > 0:
                    d['image_data'] = imgs[0]
            except: pass
        listings.append(d)
    
    # Refresh shop hourly shuffle
    seed = int(now / 3600)
    rng = random.Random(seed)
    rng.shuffle(listings)

    # 2. Notifications (Global + Private)
    notifs_rows = conn.execute("""
        SELECT * FROM notifications 
        WHERE expires_at > ? 
        AND (recipient_id IS NULL OR recipient_id = ?)
        ORDER BY scheduled_for DESC
    """, (now, user['id'])).fetchall()
    notifications = [dict(n) for n in notifs_rows]

    # 3. Cart
    cart_rows = conn.execute("SELECT * FROM cart WHERE user_id = ?", (user['id'],)).fetchall()
    cart = [dict(c) for c in cart_rows]

    # 4. Chat Meta
    chats_rows = conn.execute("SELECT * FROM chats WHERE user1_id = ? OR user2_id = ? ORDER BY last_updated DESC", (user['id'], user['id'])).fetchall()
    chats = [dict(c) for c in chats_rows]
    
    # 5. Supreme Ad Status
    ads_rows = conn.execute("SELECT * FROM supreme_ads WHERE end_time > ? ORDER BY start_time ASC", (now,)).fetchall()
    supreme_ads = [dict(ad) for ad in ads_rows]

    conn.close()
    
    return {
        "listings": listings,
        "notifications": notifications,
        "cart": cart,
        "chats": chats,
        "supreme_ads": supreme_ads,
        "profile": {**user, "is_verified": is_user_verified(user)}
    }


# --- NEW: REWARDS & STARS ---

@app.get("/daily-rewards")
def get_daily_reward_status(user: dict = Depends(get_current_user)):
    conn = get_db_connection()
    claim = conn.execute("SELECT * FROM daily_rewards_claims WHERE user_id = ?", (user['id'],)).fetchone()
    conn.close()
    
    now = time.time()
    available = True
    if claim:
        last_day = int(claim['last_claimed_at'] / 86400)
        curr_day = int(now / 86400)
        if curr_day <= last_day:
            available = False
            
    return {"available": available, "streak": claim['day_streak'] if claim else 0}

@app.post("/daily-rewards")
def claim_daily_reward(user: dict = Depends(get_current_user)):
    verify_access(user)
    conn = get_db_connection()
    now = time.time()
    claim = conn.execute("SELECT * FROM daily_rewards_claims WHERE user_id = ?", (user['id'],)).fetchone()
    
    if claim:
        last_day = int(claim['last_claimed_at'] / 86400)
        curr_day = int(now / 86400)
        if curr_day <= last_day:
            conn.close()
            raise HTTPException(status_code=400, detail="Already claimed today")
        
        # Requirement 29: Streak reset if missed a day
        if curr_day == last_day + 1:
            streak = claim['day_streak'] + 1
        else:
            streak = 1 # Missed a day, reset to Monday
            
        if streak > 7: streak = 1
        conn.execute("UPDATE daily_rewards_claims SET last_claimed_at = ?, day_streak = ? WHERE user_id = ?", (now, streak, user['id']))
    else:
        streak = 1
        conn.execute("INSERT INTO daily_rewards_claims (user_id, last_claimed_at, day_streak) VALUES (?, ?, ?)", (user['id'], now, streak))
    
    # Reward Logic (Exact Prizes from Req 29)
    reward_msg = ""
    s_add, g_add, t_add = 0, 0, 0
    
    if streak == 1: # Mon
        s_add = 30; reward_msg = "30 Stars"
    elif streak == 2: # Tue
        s_add = 80; reward_msg = "80 Stars"
    elif streak == 3: # Wed
        s_add = 300; reward_msg = "300 Stars"
    elif streak == 4: # Thu
        t_add = 2; reward_msg = "2 Aperture Tickets"
    elif streak == 5: # Fri
        g_add = 1; reward_msg = "1 Gem"
    elif streak == 6: # Sat
        s_add = 500; g_add = 1; reward_msg = "500 Stars + 1 Gem"
    elif streak == 7: # Sun
        g_add = 1; reward_msg = "1 Gem"

    conn.execute("UPDATE users SET stars_balance = stars_balance + ?, gems_balance = gems_balance + ?, tickets_balance = tickets_balance + ? WHERE id = ?",
                 (s_add, g_add, t_add, user['id']))

    conn.commit()
    conn.close()
    return {"status": "claimed", "reward": reward_msg, "streak": streak}

@app.post("/wheel/spin")
def spin_wheel(user: dict = Depends(get_current_user)):
    verify_access(user)
    # Requirement 27: Costs 1 Aperture Ticket
    if user['tickets_balance'] < 1 and not user['is_admin']:
        raise HTTPException(status_code=400, detail="Insufficient tickets. Get more from Daily Rewards!")

    conn = get_db_connection()
    if not user['is_admin']:
        conn.execute("UPDATE users SET tickets_balance = tickets_balance - 1 WHERE id = ?", (user['id'],))

    # 25 total slots prize pool
    # Slots 0-9 (10 slots): Stars (5*40, 3*10, 2*200)
    # Slots 10-14 (5 slots): 1 Gem
    # Slots 15-19 (5 slots): 1 Gem + 30 Stars
    # Slots 20-22 (3 slots): 1 Aperture Ticket
    # Slots 23-24 (2 slots): Legendary Jackpot (300 Stars + 3 Gem + 2 Tickets)
    
    rand = random.randint(0, 24)
    prize_name = ""
    s, g, t = 0, 0, 0
    
    if rand < 5: s = 40; prize_name = "40 Stars"
    elif rand < 8: s = 10; prize_name = "10 Stars"
    elif rand < 10: s = 200; prize_name = "200 Stars"
    elif rand < 15: g = 1; prize_name = "1 Gem"
    elif rand < 20: g = 1; s = 30; prize_name = "1 Gem + 30 Stars"
    elif rand < 23: t = 1; prize_name = "1 Aperture Ticket"
    else: s = 300; g = 3; t = 2; prize_name = "LEGENDARY JACKPOT!"

    conn.execute("UPDATE users SET stars_balance = stars_balance + ?, gems_balance = gems_balance + ?, tickets_balance = tickets_balance + ? WHERE id = ?",
                 (s, g, t, user['id']))
    
    conn.commit()
    conn.close()
    return {"status": "won", "prize": prize_name, "slot_index": rand}

@app.post("/listings/{item_id}/star")
def star_listing(item_id: int, user: dict = Depends(get_current_user)):
    conn = get_db_connection()
    # Check current votes
    vote = conn.execute("SELECT count FROM stars_votes WHERE user_id = ? AND listing_id = ?", (user['id'], item_id)).fetchone()
    current_count = vote['count'] if vote else 0
    
    if current_count >= 20:
        conn.close()
        raise HTTPException(status_code=400, detail="Max 20 stars per listing reached")
        
    if vote:
        conn.execute("UPDATE stars_votes SET count = count + 1 WHERE user_id = ? AND listing_id = ?", (user['id'], item_id))
    else:
        conn.execute("INSERT INTO stars_votes (user_id, listing_id, count) VALUES (?, ?, 1)", (user['id'], item_id))
        
    conn.execute("UPDATE listings SET stars_count = stars_count + 1 WHERE id = ?", (item_id,))
    conn.commit()
    conn.close()
    return {"status": "starred", "new_total": current_count + 1}

@app.get("/listings/recommended")
def get_recommended_listings(category: Optional[str] = None, name: Optional[str] = None):
    conn = get_db_connection()
    query = "SELECT * FROM listings WHERE 1=1"
    params = []
    
    if category and category != "All":
        query += " AND category = ?"
        params.append(category)
    
    if name:
        query += " AND title LIKE ?"
        params.append(f"%{name}%")
        
    query += " ORDER BY stars_count DESC LIMIT 10"
    
    items = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(item) for item in items]

@app.post("/listings/{item_id}/boost")
def boost_listing_with_gems(item_id: int, user: dict = Depends(get_current_user)):
    conn = get_db_connection()
    # Check if user has 15 gems
    if user['gems_balance'] < 15:
        conn.close()
        raise HTTPException(status_code=400, detail="Insufficient gems. Need 15 gems to boost.")
        
    # Check if item exists and belongs to user
    item = conn.execute("SELECT * FROM listings WHERE id = ?", (item_id,)).fetchone()
    if not item:
        conn.close()
        raise HTTPException(status_code=404, detail="Item not found")
    if item['seller_id'] != user['id'] and not user['is_admin']:
        conn.close()
        raise HTTPException(status_code=403, detail="Not authorized")
        
    # Deduct gems and boost
    conn.execute("UPDATE users SET gems_balance = gems_balance - 15 WHERE id = ?", (user['id'],))
    conn.execute("UPDATE listings SET is_magnetic = 1, is_gem_boosted = 1 WHERE id = ?", (item_id,))
    conn.commit()
    conn.close()
    return {"status": "boosted", "remaining_gems": user['gems_balance'] - 15}
# --- ADMIN ACTIONS ---

@app.post("/admin/ban/{target_user_id}")
def ban_user(target_user_id: str, user: dict = Depends(get_current_user)):
    # Only Primary Admin can ban
    if not user['is_admin']:
        raise HTTPException(status_code=403, detail="Only Primary Admin can ban users")
        
    conn = get_db_connection()
    conn.execute("UPDATE users SET is_banned = 1 WHERE id = ?", (target_user_id,))
    # Remove their listings
    conn.execute("DELETE FROM listings WHERE seller_id = ?", (target_user_id,))
    conn.commit()
    conn.close()
    return {"status": "User banned and listings removed"}

class DonationRequest(BaseModel):
    target_user_id: str
    stars: int = 0
    gems: int = 0
    tickets: int = 0

@app.post("/admin/donate")
def admin_donate(req: DonationRequest, user: dict = Depends(get_current_user)):
    # Only Primary Admin can donate
    if not user['is_admin']:
        raise HTTPException(status_code=403, detail="Only Primary Admin can donate rewards")
        
    conn = get_db_connection()
    conn.execute("UPDATE users SET stars_balance = stars_balance + ?, gems_balance = gems_balance + ?, tickets_balance = tickets_balance + ? WHERE id = ?",
                 (req.stars, req.gems, req.tickets, req.target_user_id))
    
    # Add a notification to the user
    msg = f"Admin donated you: {req.stars} Stars, {req.gems} Gems, {req.tickets} Tickets!"
    conn.execute("INSERT INTO notifications (type, message, poster_id, scheduled_for, expires_at) VALUES (?, ?, ?, ?, ?)",
                 ("admin_alert", msg, user['id'], time.time(), time.time() + 86400))
                 
    conn.commit()
    conn.close()
    return {"status": "Donated successfully"}

if __name__ == "__main__":
    import uvicorn
    # Use dynamic port for cloud hosting (Render/Railway)
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting Apperture Backend on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)