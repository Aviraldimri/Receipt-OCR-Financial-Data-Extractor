"""
generate_receipts.py — Sample Receipt Image Generator
======================================================
Creates realistic receipt images with varying conditions:
- Different store layouts
- Noise, blur, skew
- Varying lighting/contrast
"""

import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import cv2


# Receipt data templates
RECEIPT_TEMPLATES = [
    {
        "store_name": "FRESH MART GROCERY",
        "address": "123 Main Street, Springfield",
        "phone": "Tel: (555) 123-4567",
        "date": "03/15/2025",
        "time": "14:32:05",
        "cashier": "Cashier: Sarah M.",
        "items": [
            ("Organic Bananas", 2.49),
            ("Whole Milk 1 Gal", 4.99),
            ("Sourdough Bread", 5.49),
            ("Cheddar Cheese", 6.99),
            ("Chicken Breast", 8.99),
            ("Fresh Spinach", 3.49),
            ("Olive Oil 500ml", 7.99),
        ],
        "tax_rate": 0.08,
    },
    {
        "store_name": "TECH WORLD ELECTRONICS",
        "address": "456 Oak Avenue, Downtown Plaza",
        "phone": "Phone: (555) 987-6543",
        "date": "01/22/2025",
        "time": "10:15:30",
        "cashier": "Associate: Mike R.",
        "items": [
            ("USB-C Cable 6ft", 12.99),
            ("Wireless Mouse", 29.99),
            ("Screen Protector", 9.99),
            ("Phone Case", 19.99),
            ("Earbuds Pro", 49.99),
        ],
        "tax_rate": 0.09,
    },
    {
        "store_name": "SUNNY CAFE & BAKERY",
        "address": "789 Elm Street, Riverside",
        "phone": "Tel: (555) 456-7890",
        "date": "02/28/2025",
        "time": "08:45:12",
        "cashier": "Server: Amy L.",
        "items": [
            ("Cappuccino Large", 5.50),
            ("Blueberry Muffin", 3.75),
            ("Avocado Toast", 9.95),
            ("Fresh OJ", 4.25),
            ("Croissant", 3.50),
            ("Latte Medium", 4.75),
        ],
        "tax_rate": 0.07,
    },
    {
        "store_name": "QUICK STOP PHARMACY",
        "address": "321 Pine Road, Suite 100",
        "phone": "Phone: (555) 321-0987",
        "date": "04/10/2025",
        "time": "16:22:45",
        "cashier": "Cashier: Tom B.",
        "items": [
            ("Vitamin D3 60ct", 12.99),
            ("Hand Sanitizer", 4.49),
            ("Pain Relief 50ct", 8.99),
            ("Bandages Box", 5.99),
            ("Tissue Pack 3pk", 6.49),
        ],
        "tax_rate": 0.06,
    },
    {
        "store_name": "BOOK HAVEN",
        "address": "555 Library Lane",
        "phone": "Tel: (555) 222-3333",
        "date": "12/05/2024",
        "time": "13:10:55",
        "cashier": "Staff: Jessica W.",
        "items": [
            ("Python Cookbook", 39.99),
            ("AI Fundamentals", 49.99),
            ("Data Science 101", 34.99),
            ("Notebook Pack", 8.99),
            ("Bookmark Set", 4.99),
        ],
        "tax_rate": 0.05,
    },
]


def create_receipt_image(template: dict, width: int = 600, 
                         add_noise: bool = False,
                         add_blur: bool = False,
                         add_skew: bool = False,
                         vary_contrast: bool = False) -> Image.Image:
    """
    Generate a synthetic receipt image from a template.
    
    Args:
        template: Receipt data dictionary.
        width: Image width in pixels.
        add_noise: Add random noise to simulate poor scan quality.
        add_blur: Add Gaussian blur to simulate camera blur.
        add_skew: Rotate image slightly to simulate hand-held capture.
        vary_contrast: Vary brightness/contrast to simulate lighting issues.
    
    Returns:
        PIL Image of the receipt.
    """
    # Calculate height based on content
    num_items = len(template["items"])
    height = 450 + num_items * 50 + 300  # header + items + footer
    
    # Create white image
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Use a default font (monospace-style for receipts) — large sizes for OCR
    try:
        font_large = ImageFont.truetype("consola.ttf", 28)
        font_medium = ImageFont.truetype("consola.ttf", 22)
        font_small = ImageFont.truetype("consola.ttf", 18)
    except (OSError, IOError):
        try:
            font_large = ImageFont.truetype("cour.ttf", 28)
            font_medium = ImageFont.truetype("cour.ttf", 22)
            font_small = ImageFont.truetype("cour.ttf", 18)
        except (OSError, IOError):
            font_large = ImageFont.load_default()
            font_medium = ImageFont.load_default()
            font_small = ImageFont.load_default()
    
    y = 30
    margin = 30
    line_width = width - 2 * margin
    
    # ─── HEADER ───
    # Store name (centered, large)
    store_name = template["store_name"]
    bbox = draw.textbbox((0, 0), store_name, font=font_large)
    text_w = bbox[2] - bbox[0]
    draw.text(((width - text_w) // 2, y), store_name, fill=(0, 0, 0), font=font_large)
    y += 40
    
    # Address
    address = template["address"]
    bbox = draw.textbbox((0, 0), address, font=font_small)
    text_w = bbox[2] - bbox[0]
    draw.text(((width - text_w) // 2, y), address, fill=(0, 0, 0), font=font_small)
    y += 28
    
    # Phone
    phone = template["phone"]
    bbox = draw.textbbox((0, 0), phone, font=font_small)
    text_w = bbox[2] - bbox[0]
    draw.text(((width - text_w) // 2, y), phone, fill=(0, 0, 0), font=font_small)
    y += 32
    
    # Separator
    draw.line([(margin, y), (width - margin, y)], fill=(0, 0, 0), width=1)
    y += 10
    
    # Date and Time
    date_time = f"Date: {template['date']}   Time: {template['time']}"
    draw.text((margin, y), date_time, fill=(0, 0, 0), font=font_small)
    y += 28
    
    # Cashier
    draw.text((margin, y), template["cashier"], fill=(0, 0, 0), font=font_small)
    y += 32
    
    # Separator
    draw.line([(margin, y), (width - margin, y)], fill=(0, 0, 0), width=1)
    y += 10
    
    # Column headers
    draw.text((margin, y), "ITEM", fill=(0, 0, 0), font=font_medium)
    draw.text((width - margin - 70, y), "PRICE", fill=(0, 0, 0), font=font_medium)
    y += 30
    draw.line([(margin, y), (width - margin, y)], fill=(100, 100, 100), width=1)
    y += 12
    
    # ─── ITEMS ───
    subtotal = 0.0
    for item_name, price in template["items"]:
        # Item name (left-aligned)
        draw.text((margin, y), item_name, fill=(0, 0, 0), font=font_medium)
        # Price (right-aligned)
        price_str = f"${price:.2f}"
        bbox = draw.textbbox((0, 0), price_str, font=font_medium)
        price_w = bbox[2] - bbox[0]
        draw.text((width - margin - price_w, y), price_str, fill=(0, 0, 0), font=font_medium)
        y += 38
        subtotal += price
    
    y += 5
    draw.line([(margin, y), (width - margin, y)], fill=(0, 0, 0), width=1)
    y += 10
    
    # ─── TOTALS ───
    # Subtotal
    subtotal_text = "Subtotal:"
    subtotal_val = f"${subtotal:.2f}"
    draw.text((margin, y), subtotal_text, fill=(0, 0, 0), font=font_medium)
    bbox = draw.textbbox((0, 0), subtotal_val, font=font_medium)
    val_w = bbox[2] - bbox[0]
    draw.text((width - margin - val_w, y), subtotal_val, fill=(0, 0, 0), font=font_medium)
    y += 32
    
    # Tax
    tax = subtotal * template["tax_rate"]
    tax_text = f"Tax ({template['tax_rate']*100:.0f}%):"
    tax_val = f"${tax:.2f}"
    draw.text((margin, y), tax_text, fill=(0, 0, 0), font=font_medium)
    bbox = draw.textbbox((0, 0), tax_val, font=font_medium)
    val_w = bbox[2] - bbox[0]
    draw.text((width - margin - val_w, y), tax_val, fill=(0, 0, 0), font=font_medium)
    y += 32
    
    # Separator
    draw.line([(margin, y), (width - margin, y)], fill=(0, 0, 0), width=2)
    y += 8
    
    # Total
    total = subtotal + tax
    total_text = "TOTAL:"
    total_val = f"${total:.2f}"
    draw.text((margin, y), total_text, fill=(0, 0, 0), font=font_large)
    bbox = draw.textbbox((0, 0), total_val, font=font_large)
    val_w = bbox[2] - bbox[0]
    draw.text((width - margin - val_w, y), total_val, fill=(0, 0, 0), font=font_large)
    y += 45
    
    draw.line([(margin, y), (width - margin, y)], fill=(0, 0, 0), width=2)
    y += 15
    
    # ─── FOOTER ───
    payment = random.choice(["VISA ****1234", "CASH", "MASTERCARD ****5678", "DEBIT ****9012"])
    draw.text((margin, y), f"Payment: {payment}", fill=(0, 0, 0), font=font_small)
    y += 30
    
    thank_you = "Thank you for shopping!"
    bbox = draw.textbbox((0, 0), thank_you, font=font_medium)
    text_w = bbox[2] - bbox[0]
    draw.text(((width - text_w) // 2, y), thank_you, fill=(0, 0, 0), font=font_medium)
    y += 35
    
    ref_num = f"Ref #: {random.randint(100000, 999999)}"
    bbox = draw.textbbox((0, 0), ref_num, font=font_small)
    text_w = bbox[2] - bbox[0]
    draw.text(((width - text_w) // 2, y), ref_num, fill=(0, 0, 0), font=font_small)
    y += 35
    
    # Crop to content
    img = img.crop((0, 0, width, min(y + 20, height)))
    
    # ─── POST-PROCESSING EFFECTS ───
    
    # Convert to numpy for OpenCV operations
    img_np = np.array(img)
    
    if vary_contrast:
        # Randomly adjust brightness and contrast
        alpha = random.uniform(0.75, 1.15)  # Contrast
        beta = random.randint(-25, 25)     # Brightness
        img_np = cv2.convertScaleAbs(img_np, alpha=alpha, beta=beta)
    
    if add_noise:
        # Add mild Gaussian noise
        noise = np.random.normal(0, random.randint(5, 15), img_np.shape).astype(np.uint8)
        img_np = cv2.add(img_np, noise)
    
    if add_blur:
        # Add Gaussian blur
        kernel = random.choice([3, 5])
        img_np = cv2.GaussianBlur(img_np, (kernel, kernel), 0)
    
    img = Image.fromarray(img_np)
    
    if add_skew:
        # Rotate by a small random angle
        angle = random.uniform(-3, 3)
        img = img.rotate(angle, expand=True, fillcolor=(245, 245, 245))
    
    return img


def generate_sample_receipts(output_dir: str = "sample_receipts", count: int = 5) -> list:
    """
    Generate sample receipt images with varying conditions.
    
    Args:
        output_dir: Directory to save generated images.
        count: Number of receipts to generate.
    
    Returns:
        List of generated file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    generated_files = []
    
    conditions = [
        {"add_noise": False, "add_blur": False, "add_skew": False, "vary_contrast": False},  # Clean
        {"add_noise": True,  "add_blur": False, "add_skew": False, "vary_contrast": False},  # Noisy
        {"add_noise": False, "add_blur": True,  "add_skew": True,  "vary_contrast": False},  # Blurry + skewed
        {"add_noise": True,  "add_blur": True,  "add_skew": False, "vary_contrast": True},   # Noisy + blur + contrast
        {"add_noise": True,  "add_blur": False, "add_skew": True,  "vary_contrast": True},   # All effects
    ]
    
    for i in range(min(count, len(RECEIPT_TEMPLATES))):
        template = RECEIPT_TEMPLATES[i]
        condition = conditions[i % len(conditions)]
        
        img = create_receipt_image(template, **condition)
        
        filename = f"receipt_{i+1}.png"
        filepath = os.path.join(output_dir, filename)
        img.save(filepath, "PNG")
        generated_files.append(filepath)
        
        condition_desc = []
        if condition["add_noise"]: condition_desc.append("noisy")
        if condition["add_blur"]: condition_desc.append("blurry")
        if condition["add_skew"]: condition_desc.append("skewed")
        if condition["vary_contrast"]: condition_desc.append("varied contrast")
        if not condition_desc: condition_desc.append("clean")
        
        print(f"  Generated {filename} ({', '.join(condition_desc)})")
    
    return generated_files


if __name__ == "__main__":
    print("Generating sample receipt images...\n")
    files = generate_sample_receipts()
    print(f"\nGenerated {len(files)} receipt images in 'sample_receipts/' directory")
