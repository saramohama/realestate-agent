from flask import Flask, request, jsonify
import anthropic
import json
import os
from datetime import datetime

app = Flask(__name__)

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# ─────────────────────────────────────────────
# AGENT CONTEXT
# ─────────────────────────────────────────────

AGENCY_INFO = """
Agency Name: Prestige Realty Group
Location: 456 Park Avenue, New York, NY
Phone: 555-9000
Email: info@prestigerealty.com
Hours: Mon-Fri 9am-6pm, Sat 10am-4pm

PROPERTY TYPES WE HANDLE:
- Apartments / Condos
- Single-Family Homes
- Townhouses
- Commercial Properties

SERVICES:
- Property listing & marketing
- Buyer consultation & tours
- Mortgage calculator assistance
- Viewing scheduling
"""

SYSTEM_PROMPT = f"""
You are Alex, a professional real estate listing assistant for Prestige Realty Group.
Your job is to help clients list properties, search available listings, schedule viewings, and calculate mortgage estimates.

Keep responses SHORT and professional.
Use plain text — no markdown symbols like ** or ##.
Always be warm, helpful, and guide the user step by step.

When a client wants to LIST a property, collect:
  - Owner name
  - Property address
  - Property type (apartment, house, condo, commercial)
  - Bedrooms & bathrooms
  - Square footage
  - Asking price
  - Description / key features

When a client wants to SCHEDULE a viewing, collect:
  - Client name
  - Property address or listing ID
  - Preferred date and time
  - Contact phone number

When asked to CALCULATE MORTGAGE, collect:
  - Property price
  - Down payment amount
  - Loan term (years)
  - Interest rate (or use 7% as default)

{AGENCY_INFO}
"""

# ─────────────────────────────────────────────
# IN-MEMORY STORE  (replace with a DB in prod)
# ─────────────────────────────────────────────

conversations: dict = {}

# ─────────────────────────────────────────────
# TOOL IMPLEMENTATIONS
# ─────────────────────────────────────────────

def save_listing(owner_name, address, property_type, bedrooms, bathrooms,
                 sqft, price, description):
    listing = {
        "listing_id": "LST-" + datetime.now().strftime("%Y%m%d%H%M%S"),
        "owner_name": owner_name,
        "address": address,
        "property_type": property_type,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "sqft": sqft,
        "price": price,
        "description": description,
        "status": "active",
        "listed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    _append_json("listings.json", listing)
    return f"Listing saved! ID: {listing['listing_id']} — {address} listed at {price}"


def search_listings(property_type=None, max_price=None, min_bedrooms=None, location=None):
    try:
        with open("listings.json", "r") as f:
            all_listings = json.load(f)
    except FileNotFoundError:
        return "No listings found in the database yet."

    results = [l for l in all_listings if l.get("status") == "active"]

    if property_type:
        results = [l for l in results if property_type.lower() in l.get("property_type", "").lower()]
    if max_price:
        def parse_price(p):
            return float(str(p).replace("$", "").replace(",", "").strip())
        try:
            results = [l for l in results if parse_price(l.get("price", "0")) <= float(max_price)]
        except ValueError:
            pass
    if min_bedrooms:
        try:
            results = [l for l in results if int(l.get("bedrooms", 0)) >= int(min_bedrooms)]
        except ValueError:
            pass
    if location:
        results = [l for l in results if location.lower() in l.get("address", "").lower()]

    if not results:
        return "No listings match your criteria."

    lines = [f"Found {len(results)} listing(s):"]
    for l in results[:5]:   # cap at 5 for readability
        lines.append(
            f"- [{l['listing_id']}] {l['address']} | {l['property_type']} | "
            f"{l['bedrooms']}bd/{l['bathrooms']}ba | {l['sqft']} sqft | {l['price']}"
        )
    return "\n".join(lines)


def schedule_viewing(client_name, property_address, preferred_datetime, phone):
    viewing = {
        "viewing_id": "VW-" + datetime.now().strftime("%Y%m%d%H%M%S"),
        "client_name": client_name,
        "property_address": property_address,
        "preferred_datetime": preferred_datetime,
        "phone": phone,
        "status": "scheduled",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    _append_json("viewings.json", viewing)
    return (f"Viewing scheduled! ID: {viewing['viewing_id']} — "
            f"{client_name} will visit {property_address} on {preferred_datetime}. "
            f"Confirmation will be sent to {phone}.")


def calculate_mortgage(property_price, down_payment, loan_term_years, annual_interest_rate=7.0):
    try:
        price = float(str(property_price).replace("$", "").replace(",", ""))
        down  = float(str(down_payment).replace("$", "").replace(",", ""))
        term  = int(loan_term_years)
        rate  = float(annual_interest_rate)

        principal = price - down
        monthly_rate = rate / 100 / 12
        n_payments = term * 12

        if monthly_rate == 0:
            monthly = principal / n_payments
        else:
            monthly = principal * (monthly_rate * (1 + monthly_rate) ** n_payments) / \
                      ((1 + monthly_rate) ** n_payments - 1)

        total_paid = monthly * n_payments
        total_interest = total_paid - principal

        return (
            f"Mortgage Estimate:\n"
            f"  Property Price:   ${price:,.2f}\n"
            f"  Down Payment:     ${down:,.2f}\n"
            f"  Loan Amount:      ${principal:,.2f}\n"
            f"  Interest Rate:    {rate}% / year\n"
            f"  Loan Term:        {term} years\n"
            f"  Monthly Payment:  ${monthly:,.2f}\n"
            f"  Total Interest:   ${total_interest:,.2f}\n"
            f"  Total Cost:       ${total_paid + down:,.2f}"
        )
    except Exception as e:
        return f"Could not calculate mortgage: {str(e)}"


# ─────────────────────────────────────────────
# TOOL DEFINITIONS  
# ─────────────────────────────────────────────

TOOLS = [
    {
        "name": "save_listing",
        "description": "Save a new property listing when the owner has provided all details.",
        "input_schema": {
            "type": "object",
            "properties": {
                "owner_name":     {"type": "string", "description": "Full name of the property owner"},
                "address":        {"type": "string", "description": "Full property address"},
                "property_type":  {"type": "string", "description": "Type: apartment, house, condo, commercial"},
                "bedrooms":       {"type": "string", "description": "Number of bedrooms"},
                "bathrooms":      {"type": "string", "description": "Number of bathrooms"},
                "sqft":           {"type": "string", "description": "Square footage"},
                "price":          {"type": "string", "description": "Asking price e.g. $450,000"},
                "description":    {"type": "string", "description": "Key features and description"},
            },
            "required": ["owner_name", "address", "property_type", "bedrooms",
                         "bathrooms", "sqft", "price", "description"],
        },
    },
    {
        "name": "search_listings",
        "description": "Search active property listings by filters.",
        "input_schema": {
            "type": "object",
            "properties": {
                "property_type":  {"type": "string", "description": "Filter by type (optional)"},
                "max_price":      {"type": "string", "description": "Maximum price filter (optional)"},
                "min_bedrooms":   {"type": "string", "description": "Minimum bedrooms (optional)"},
                "location":       {"type": "string", "description": "Location keyword (optional)"},
            },
            "required": [],
        },
    },
    {
        "name": "schedule_viewing",
        "description": "Schedule a property viewing for a client.",
        "input_schema": {
            "type": "object",
            "properties": {
                "client_name":         {"type": "string", "description": "Client's full name"},
                "property_address":    {"type": "string", "description": "Address or listing ID of the property"},
                "preferred_datetime":  {"type": "string", "description": "Preferred date and time for the viewing"},
                "phone":               {"type": "string", "description": "Client's contact phone number"},
            },
            "required": ["client_name", "property_address", "preferred_datetime", "phone"],
        },
    },
    {
        "name": "calculate_mortgage",
        "description": "Calculate estimated monthly mortgage payment.",
        "input_schema": {
            "type": "object",
            "properties": {
                "property_price":      {"type": "string", "description": "Total property price"},
                "down_payment":        {"type": "string", "description": "Down payment amount"},
                "loan_term_years":     {"type": "string", "description": "Loan term in years (e.g. 30)"},
                "annual_interest_rate":{"type": "string", "description": "Annual interest rate % (default 7.0)"},
            },
            "required": ["property_price", "down_payment", "loan_term_years"],
        },
    },
]

# ─────────────────────────────────────────────
# TOOL DISPATCHER
# ─────────────────────────────────────────────

TOOL_MAP = {
    "save_listing":      save_listing,
    "search_listings":   search_listings,
    "schedule_viewing":  schedule_viewing,
    "calculate_mortgage": calculate_mortgage,
}

def dispatch_tool(name, inputs):
    fn = TOOL_MAP.get(name)
    if fn:
        return fn(**inputs)
    return f"Unknown tool: {name}"

# ─────────────────────────────────────────────
# AGENT LOOP
# ─────────────────────────────────────────────

def get_agent_reply(session_id: str, user_message: str) -> str:
    if session_id not in conversations:
        conversations[session_id] = []

    conversations[session_id].append({"role": "user", "content": user_message})

    while True:
        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=1000,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=conversations[session_id],
        )

        if response.stop_reason == "tool_use":
            # Append assistant turn (may contain text + tool_use blocks)
            conversations[session_id].append({
                "role": "assistant",
                "content": response.content,
            })
            # Process every tool call and build tool_result turn
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = dispatch_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })
            conversations[session_id].append({
                "role": "user",
                "content": tool_results,
            })
            # Loop — let Claude produce the final reply

        else:  # end_turn
            reply = next(
                (b.text for b in response.content if hasattr(b, "text")), ""
            )
            conversations[session_id].append({"role": "assistant", "content": reply})
            return reply

# ─────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────

def _append_json(filename: str, record: dict):
    try:
        with open(filename, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        data = []
    data.append(record)
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

# ─────────────────────────────────────────────
# FLASK ROUTES
# ─────────────────────────────────────────────

@app.route("/chat", methods=["POST"])
def chat():
    """
    Expects JSON body:
        { "session_id": "user-123", "message": "I want to list my apartment" }
    Returns:
        { "reply": "...", "session_id": "user-123" }
    """
    body = request.get_json(force=True)
    session_id = body.get("session_id", "default")
    message    = body.get("message", "")

    if not message:
        return jsonify({"error": "message is required"}), 400

    reply = get_agent_reply(session_id, message)
    return jsonify({"reply": reply, "session_id": session_id})


@app.route("/listings", methods=["GET"])
def list_all():
    """Return all saved listings."""
    try:
        with open("listings.json", "r") as f:
            return jsonify(json.load(f))
    except FileNotFoundError:
        return jsonify([])


@app.route("/viewings", methods=["GET"])
def list_viewings():
    """Return all scheduled viewings."""
    try:
        with open("viewings.json", "r") as f:
            return jsonify(json.load(f))
    except FileNotFoundError:
        return jsonify([])


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "agent": "Prestige Realty Group Assistant"})


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("Real Estate Agent running on http://localhost:5000")
    print("Endpoints:")
    print("  POST /chat       — talk to the agent")
    print("  GET  /listings   — view all listings")
    print("  GET  /viewings   — view all scheduled viewings")
    print("  GET  /health     — health check")
    app.run(debug=False, port=5000)
