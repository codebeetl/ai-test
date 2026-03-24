"""Initialisation script — pre-seeds all data stores with realistic test data.

Run once before first use:
    python scripts/init_data.py
"""

import sys
import sqlite3
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.config.settings import load_settings
from src.memory.golden_bucket import GoldenBucket, Trio
from src.memory.user_prefs import UserPrefsStore

settings = load_settings()


EXPERT_TRIOS = [
    Trio(
        question="What are the top 10 products by total revenue?",
        sql="""SELECT
  p.name AS product_name,
  p.category,
  ROUND(SUM(oi.sale_price), 2) AS total_revenue
FROM `bigquery-public-data.thelook_ecommerce.order_items` oi
JOIN `bigquery-public-data.thelook_ecommerce.products` p ON oi.product_id = p.id
WHERE oi.status NOT IN ('Cancelled', 'Returned')
GROUP BY p.name, p.category
ORDER BY total_revenue DESC
LIMIT 10""",
        report="Top 10 products by revenue excluding cancellations and returns.",
    ),
    Trio(
        question="What is the total number of orders and revenue per month this year?",
        sql="""SELECT
  FORMAT_DATE('%Y-%m', DATE(created_at)) AS month,
  COUNT(DISTINCT order_id) AS total_orders,
  ROUND(SUM(sale_price), 2) AS total_revenue
FROM `bigquery-public-data.thelook_ecommerce.order_items`
WHERE DATE(created_at) >= DATE_TRUNC(CURRENT_DATE(), YEAR)
  AND status NOT IN ('Cancelled', 'Returned')
GROUP BY month
ORDER BY month""",
        report="Monthly order counts and revenue for the current calendar year.",
    ),
    Trio(
        question="Who are the top 10 customers by total spend?",
        sql="""SELECT
  u.id AS user_id,
  u.first_name,
  u.last_name,
  u.country,
  COUNT(DISTINCT o.order_id) AS total_orders,
  ROUND(SUM(oi.sale_price), 2) AS total_spend
FROM `bigquery-public-data.thelook_ecommerce.users` u
JOIN `bigquery-public-data.thelook_ecommerce.orders` o ON u.id = o.user_id
JOIN `bigquery-public-data.thelook_ecommerce.order_items` oi ON o.order_id = oi.order_id
WHERE oi.status NOT IN ('Cancelled', 'Returned')
GROUP BY u.id, u.first_name, u.last_name, u.country
ORDER BY total_spend DESC
LIMIT 10""",
        report="Top 10 customers by lifetime spend with order count and country.",
    ),
    Trio(
        question="What is the return rate by product category?",
        sql="""SELECT
  p.category,
  COUNT(*) AS total_items,
  COUNTIF(oi.status = 'Returned') AS returned_items,
  ROUND(COUNTIF(oi.status = 'Returned') / COUNT(*) * 100, 2) AS return_rate_pct
FROM `bigquery-public-data.thelook_ecommerce.order_items` oi
JOIN `bigquery-public-data.thelook_ecommerce.products` p ON oi.product_id = p.id
GROUP BY p.category
ORDER BY return_rate_pct DESC""",
        report="Return rate percentage per product category, highest first.",
    ),
    Trio(
        question="How many new customers signed up each month this year?",
        sql="""SELECT
  FORMAT_DATE('%Y-%m', DATE(created_at)) AS month,
  COUNT(*) AS new_customers
FROM `bigquery-public-data.thelook_ecommerce.users`
WHERE DATE(created_at) >= DATE_TRUNC(CURRENT_DATE(), YEAR)
GROUP BY month
ORDER BY month""",
        report="Monthly new customer acquisition count for the current year.",
    ),
    Trio(
        question="What is the average order value by country?",
        sql="""SELECT
  u.country,
  COUNT(DISTINCT o.order_id) AS total_orders,
  ROUND(AVG(o.num_of_item * oi.sale_price), 2) AS avg_order_value
FROM `bigquery-public-data.thelook_ecommerce.orders` o
JOIN `bigquery-public-data.thelook_ecommerce.users` u ON o.user_id = u.id
JOIN `bigquery-public-data.thelook_ecommerce.order_items` oi ON o.order_id = oi.order_id
WHERE oi.status NOT IN ('Cancelled', 'Returned')
GROUP BY u.country
ORDER BY avg_order_value DESC
LIMIT 20""",
        report="Average order value and order count by country, top 20 markets.",
    ),
    Trio(
        question="Which product categories have the highest revenue this quarter?",
        sql="""SELECT
  p.category,
  ROUND(SUM(oi.sale_price), 2) AS total_revenue,
  COUNT(DISTINCT oi.order_id) AS total_orders
FROM `bigquery-public-data.thelook_ecommerce.order_items` oi
JOIN `bigquery-public-data.thelook_ecommerce.products` p ON oi.product_id = p.id
WHERE DATE(oi.created_at) >= DATE_TRUNC(CURRENT_DATE(), QUARTER)
  AND oi.status NOT IN ('Cancelled', 'Returned')
GROUP BY p.category
ORDER BY total_revenue DESC""",
        report="Category revenue and order count for the current quarter.",
    ),
    Trio(
        question="What is the overall order completion rate versus cancellation rate?",
        sql="""SELECT
  status,
  COUNT(*) AS order_count,
  ROUND(COUNT(*) / SUM(COUNT(*)) OVER () * 100, 2) AS pct_of_total
FROM `bigquery-public-data.thelook_ecommerce.order_items`
GROUP BY status
ORDER BY order_count DESC""",
        report="Order status breakdown showing completion, cancellation, and return rates.",
    ),
]

SAVED_REPORTS = [
    ("Q1 Performance Summary — Acme Corp", "Acme Corp",
     "Acme Corp achieved £2.4M revenue in Q1, up 12% YoY. Top category: Outerwear."),
    ("Return Rate Analysis — Acme Corp", "Acme Corp",
     "Acme Corp return rate stands at 8.3%, driven by sizing issues in Jeans category."),
    ("Monthly Revenue Breakdown — Globex Ltd", "Globex Ltd",
     "Globex Ltd monthly revenue averaged £180K across Q1. Peak month: March at £210K."),
    ("Customer Acquisition Report — Globex Ltd", "Globex Ltd",
     "Globex Ltd acquired 340 new customers in Q1. Conversion rate improved to 4.2%."),
    ("Top Products Report — Initech Solutions", "Initech Solutions",
     "Initech Solutions top 5 products drove 38% of total revenue."),
    ("Annual Summary — Initech Solutions", "Initech Solutions",
     "Initech Solutions full-year revenue: £1.8M. Strong H2 offset weak Q1."),
]


def seed_golden_bucket() -> None:
    db_path = settings.memory.resolve_path(settings.memory.golden_bucket_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  [debug] Writing Golden Bucket to: {db_path}")

    gb = GoldenBucket(str(db_path), embedder=None)
    existing = gb._conn.execute("SELECT COUNT(*) FROM trios").fetchone()[0]
    if existing > 0:
        print(f"  Golden Bucket already has {existing} trios — skipping.")
        return

    gb.add_trios(EXPERT_TRIOS)
    print(f"  ✅ Golden Bucket seeded with {len(EXPERT_TRIOS)} expert trios.")


def seed_saved_reports() -> None:
    path = settings.memory.resolve_path(settings.memory.reports_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  [debug] Writing Saved Reports to: {path}")

    conn = sqlite3.connect(str(path))
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS reports (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          title TEXT NOT NULL,
          client_name TEXT,
          content TEXT NOT NULL
        )
    """)
    existing = cur.execute("SELECT COUNT(*) FROM reports").fetchone()[0]
    if existing > 0:
        print(f"  Saved Reports already has {existing} entries — skipping.")
        conn.close()
        return

    cur.executemany(
        "INSERT INTO reports (title, client_name, content) VALUES (?, ?, ?)",
        SAVED_REPORTS,
    )
    conn.commit()
    conn.close()
    print(f"  ✅ Saved Reports seeded with {len(SAVED_REPORTS)} reports across 3 clients.")
    print("     Clients: Acme Corp, Globex Ltd, Initech Solutions")
    print('     Test delete with: "Delete all reports mentioning Acme Corp"')


def seed_user_prefs() -> None:
    prefs = UserPrefsStore(
        str(settings.memory.resolve_path(settings.memory.user_prefs_path))
    )
    prefs.set_output_format("manager_a", "table")
    prefs.set_output_format("manager_b", "bullets")
    print("  ✅ User preferences set:")
    print("     manager_a → table format")
    print("     manager_b → bullet format")
    print("     Switch users with: /whoami manager_b")


if __name__ == "__main__":
    print("\n=== Retail Data Agent — Initialisation ===\n")
    print("Seeding Golden Bucket...")
    seed_golden_bucket()
    print("\nSeeding Saved Reports...")
    seed_saved_reports()
    print("\nSeeding User Preferences...")
    seed_user_prefs()
    print("\n✅ All done. Run: python main.py\n")
