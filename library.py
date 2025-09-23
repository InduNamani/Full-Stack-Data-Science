import os
from supabase import create_client, Client
from dotenv import load_dotenv
from datetime import datetime, timedelta

# Load Supabase credentials
load_dotenv()
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")
sb: Client = create_client(url, key)

# ---------------- Member Operations ----------------
def add_member(name, email):
    resp = sb.table("members").insert({"name": name, "email": email}).execute()
    return resp.data

def list_members():
    resp = sb.table("members").select("*").execute()
    return resp.data

def update_member_email(member_id, new_email):
    resp = sb.table("members").update({"email": new_email}).eq("member_id", member_id).execute()
    return resp.data

def delete_member(member_id):
    # Only allow if no borrowed books
    active_borrow = sb.table("borrow_records")\
                      .select("*").eq("member_id", member_id).is_("return_date", None).execute().data
    if active_borrow:
        print("Cannot delete: member has borrowed books not yet returned.")
        return
    sb.table("members").delete().eq("member_id", member_id).execute()
    print("Member deleted.")

def get_member_borrowed_books(member_id):
    resp = sb.table("borrow_records") \
             .select("*, books(title, author)") \
             .eq("member_id", member_id).execute()
    return resp.data

# ---------------- Book Operations ----------------
def add_book(title, author, category, stock=1):
    payload = {"title": title, "author": author, "category": category, "stock": stock}
    resp = sb.table("books").insert(payload).execute()
    return resp.data

def list_books():
    return sb.table("books").select("*").execute().data

def search_books(keyword):
    resp = sb.table("books").select("*").ilike("title", f"%{keyword}%").execute()
    return resp.data

def update_book_stock(book_id, stock):
    resp = sb.table("books").update({"stock": stock}).eq("book_id", book_id).execute()
    return resp.data

def delete_book(book_id):
    # Only allow if book is not borrowed
    active_borrow = sb.table("borrow_records")\
                      .select("*").eq("book_id", book_id).is_("return_date", None).execute().data
    if active_borrow:
        print("Cannot delete: book is currently borrowed.")
        return
    sb.table("books").delete().eq("book_id", book_id).execute()
    print("Book deleted.")

# ---------------- Borrow/Return Operations ----------------
def borrow_book(member_id, book_id):
    try:
        # Check stock
        book = sb.table("books").select("*").eq("book_id", book_id).execute().data[0]
        if book["stock"] <= 0:
            print("Book out of stock!")
            return
        # Insert borrow record
        sb.table("borrow_records").insert({"member_id": member_id, "book_id": book_id}).execute()
        # Decrease stock
        sb.table("books").update({"stock": book["stock"] - 1}).eq("book_id", book_id).execute()
        print("Book borrowed successfully!")
    except Exception as e:
        print("Transaction failed:", e)

def return_book(member_id, book_id):
    try:
        # Find active borrow record
        records = sb.table("borrow_records").select("*")\
                     .eq("member_id", member_id).eq("book_id", book_id).is_("return_date", None).execute().data
        if not records:
            print("No active borrow record found!")
            return
        record_id = records[0]["record_id"]
        # Update return_date
        sb.table("borrow_records").update({"return_date": datetime.utcnow().isoformat()})\
                                  .eq("record_id", record_id).execute()
        # Increase stock
        book = sb.table("books").select("*").eq("book_id", book_id).execute().data[0]
        sb.table("books").update({"stock": book["stock"] + 1}).eq("book_id", book_id).execute()
        print("Book returned successfully!")
    except Exception as e:
        print("Transaction failed:", e)

# ---------------- Reports ----------------
def top_borrowed_books(limit=5):
    # Aggregate borrow counts
    borrow_counts = sb.table("borrow_records").select("book_id").execute().data
    counts = {}
    for r in borrow_counts:
        counts[r["book_id"]] = counts.get(r["book_id"], 0) + 1
    sorted_books = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:limit]
    result = []
    for book_id, cnt in sorted_books:
        book = sb.table("books").select("title").eq("book_id", book_id).execute().data[0]
        result.append({"book_id": book_id, "title": book["title"], "borrow_count": cnt})
    return result

def overdue_members(days=14):
    cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
    records = sb.table("borrow_records").select("*, members(name,email), books(title)")\
                  .lt("borrow_date", cutoff).is_("return_date", None).execute().data
    return records

def books_borrowed_per_member():
    members = sb.table("members").select("*").execute().data
    result = []
    for m in members:
        count = sb.table("borrow_records").select("*").eq("member_id", m["member_id"]).execute().data
        result.append({"member_id": m["member_id"], "name": m["name"], "borrowed_count": len(count)})
    return result

# ---------------- CLI Menu ----------------
def menu():
    while True:
        print("\n=== Online Library Management ===")
        print("1. Register Member")
        print("2. Add Book")
        print("3. List Members")
        print("4. List Books")
        print("5. Search Books")
        print("6. Borrow Book")
        print("7. Return Book")
        print("8. Delete Member")
        print("9. Delete Book")
        print("10. Top Borrowed Books")
        print("11. Overdue Members")
        print("12. Borrowed Count per Member")
        print("0. Exit")
        choice = input("Enter choice: ").strip()

        if choice == "1":
            name = input("Name: ")
            email = input("Email: ")
            print(add_member(name, email))
        elif choice == "2":
            title = input("Title: ")
            author = input("Author: ")
            category = input("Category: ")
            stock = int(input("Stock: "))
            print(add_book(title, author, category, stock))
        elif choice == "3":
            print(list_members())
        elif choice == "4":
            print(list_books())
        elif choice == "5":
            keyword = input("Keyword: ")
            print(search_books(keyword))
        elif choice == "6":
            member_id = int(input("Member ID: "))
            book_id = int(input("Book ID: "))
            borrow_book(member_id, book_id)
        elif choice == "7":
            member_id = int(input("Member ID: "))
            book_id = int(input("Book ID: "))
            return_book(member_id, book_id)
        elif choice == "8":
            member_id = int(input("Member ID: "))
            delete_member(member_id)
        elif choice == "9":
            book_id = int(input("Book ID: "))
            delete_book(book_id)
        elif choice == "10":
            print(top_borrowed_books())
        elif choice == "11":
            print(overdue_members())
        elif choice == "12":
            print(books_borrowed_per_member())
        elif choice == "0":
            break
        else:
            print("Invalid choice!")

if __name__ == "__main__":
    menu()
