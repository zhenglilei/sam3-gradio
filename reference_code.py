def overlay_masks_boxes_scores(
    image,
    masks,
    boxes,
    scores,
    labels=None,
    score_threshold=0.0,
    alpha=0.5,
):
    image = image.convert("RGBA")
    masks = masks.cpu().numpy()
    boxes = boxes.cpu().numpy()
    scores = scores.cpu().numpy()
    if labels is None:
        labels = ["object"] * len(scores)
    labels = np.array(labels)
    # Score filtering
    keep = scores >= score_threshold

    masks = masks[keep]
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]
    n_instances = len(masks)
    if n_instances == 0:
        return image
    # Colormap (one color per instance)
    cmap = matplotlib.colormaps.get_cmap("rainbow").resampled(n_instances)
    colors = [
        tuple(int(c * 255) for c in cmap(i)[:3])
        for i in range(n_instances)
    ]

    # =========================
    # PASS 1: MASK OVERLAY
    # =========================
    for mask, color in zip(masks, colors):
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        overlay = Image.new("RGBA", image.size, color + (0,))
        overlay.putalpha(mask_img.point(lambda v: int(v * alpha)))
        image = Image.alpha_composite(image, overlay)

    # =========================
    # PASS 2: BOXES + LABELS
    # =========================
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for box, score, label, color in zip(boxes, scores, labels, colors):
        x1, y1, x2, y2 = map(int, box.tolist())
        # --- Bounding box (with black stroke for visibility)
        draw.rectangle([(x1, y1), (x2, y2)], outline="black", width=3)
        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=2)
        # --- Label text
        text = f"{label} | {score:.2f}"
        tb = draw.textbbox((0, 0), text, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        # Label background
        draw.rectangle(
            [(x1, y1 - th - 4), (x1 + tw + 6, y1)],
            fill=color,
        )
        # Black label text (high contrast)
        draw.text(
            (x1 + 3, y1 - th - 2),
            text,
            fill="black",
            font=font,
        )
    return image


def widget_to_sam_boxes(widget):
   boxes = []
   labels = []
   for ann in widget.bboxes:
       x = int(ann["x"])
       y = int(ann["y"])
       w = int(ann["width"])
       h = int(ann["height"])
       x1 = x
       y1 = y
       x2 = x + w
       y2 = y + h
       label = ann.get("label") or ann.get("class")
       boxes.append([x1, y1, x2, y2])
       labels.append(1 if label == "positive" else 0)
   return boxes, labels

def get_points_from_widget(widget):
   """Extract point coordinates from widget bboxes"""
   positive_points = []
   negative_points = []
 
   for ann in widget.bboxes:
       x = int(ann["x"])
       y = int(ann["y"])
       w = int(ann["width"])
       h = int(ann["height"])
     
       # Get center point of the bbox
       center_x = x + w // 2
       center_y = y + h // 2
     
       label = ann.get("label") or ann.get("class")
     
       if label == "positive":
           positive_points.append([center_x, center_y])
       elif label == "negative":
           negative_points.append([center_x, center_y])
 
   return positive_points, negative_points

def visualize_results(masks, positive_points, negative_points):
   """Display segmentation results"""
   n_masks = masks.shape[1]
 
   # Create figure with subplots
   fig, axes = plt.subplots(1, min(n_masks, 3), figsize=(15, 5))
   if n_masks == 1:
       axes = [axes]
 
   for idx in range(min(n_masks, 3)):
       mask = masks[0, idx].numpy()
     
       # Overlay mask on image
       img_array = np.array(raw_image)
       colored_mask = np.zeros_like(img_array)
       colored_mask[mask > 0] = [0, 255, 0]  # Green mask
     
       overlay = img_array.copy()
       overlay[mask > 0] = (img_array[mask > 0] * 0.5 + colored_mask[mask > 0] * 0.5).astype(np.uint8)
     
       axes[idx].imshow(overlay)
       axes[idx].set_title(f"Mask {idx + 1} (Quality Ranked)", fontsize=12, fontweight='bold')
       axes[idx].axis('off')
     
       # Plot points on each mask
       for px, py in positive_points:
           axes[idx].plot(px, py, 'go', markersize=12, markeredgecolor='white', markeredgewidth=2.5)
       for nx, ny in negative_points:
           axes[idx].plot(nx, ny, 'ro', markersize=12, markeredgecolor='white', markeredgewidth=2.5)
 
   plt.tight_layout()
   plt.show()

def segment_from_widget(b=None):
   """Run segmentation with points from widget"""
   positive_points, negative_points = get_points_from_widget(widget)
 
   if not positive_points and not negative_points:
       print("⚠️ Please add at least one point (draw small boxes on the image)!")
       return
 
   # Combine points and labels
   all_points = positive_points + negative_points
   all_labels = [1] * len(positive_points) + [0] * len(negative_points)
 
   print(f"\n🔄 Running segmentation...")
   print(f"  • {len(positive_points)} positive points: {positive_points}")
   print(f"  • {len(negative_points)} negative points: {negative_points}")
   # Prepare inputs (4D for points, 3D for labels)
   input_points = [[all_points]]  # [batch, object, points, xy]
   input_labels = [[all_labels]]   # [batch, object, labels]
 
   inputs = processor(
       images=raw_image,
       input_points=input_points,
       input_labels=input_labels,
       return_tensors="pt"
   ).to(device)
 
   # Run inference
   with torch.no_grad():
       outputs = model(**inputs)
 
   # Post-process masks
   masks = processor.post_process_masks(
       outputs.pred_masks.cpu(),
       inputs["original_sizes"]
   )[0]
 
   print(f"✅ Generated {masks.shape[1]} masks with shape {masks.shape}")
 
   # Visualize results
   visualize_results(masks, positive_points, negative_points)

def reset_widget(b=None):
   """Clear all annotations"""
   widget.bboxes = []
   print("🔄 Reset! All points cleared.")


# output.enable_custom_widget_manager()
# Load image
url = "http://images.cocodataset.org/val2017/000000136466.jpg"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
# Convert to base64
def pil_to_base64(img):
   buffer = io.BytesIO()
   img.save(buffer, format="PNG")
   return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
# Create widget
widget = BBoxWidget(
   image=pil_to_base64(image),
   classes=["positive", "negative"]
)
widget

boxes, box_labels = widget_to_sam_boxes(widget)
print("Boxes:", boxes)
print("Labels:", box_labels)

inputs = processor(
   images=image,
   input_boxes=[boxes],              # batch size = 1
   input_boxes_labels=[box_labels],
   return_tensors="pt"
).to(device)
with torch.no_grad():
   outputs = model(**inputs)
results = processor.post_process_instance_segmentation(
   outputs,
   threshold=0.5,
   mask_threshold=0.5,
   target_sizes=inputs["original_sizes"].tolist()
)[0]
print(f"Found {len(results['masks'])} objects")

labels = ["interactive object"] * len(results["scores"])
overlay_masks_boxes_scores(
   image=image,
   masks=results["masks"],
   boxes=results["boxes"],
   scores=results["scores"],
   labels=labels,
   alpha=0.45,
)

# Setup device
device = Accelerator().device
# Load model and processor
print("Loading SAM3 model...")
local_model_dir = "/data/zhenglilei/.cache/modelscope/hub/models/facebook/sam3"
model = Sam3TrackerModel.from_pretrained(local_model_dir, local_files_only=True).to(device)
processor = Sam3TrackerProcessor.from_pretrained(local_model_dir, local_files_only=True)
print("Model loaded successfully!")
# Load image
IMAGE_PATH = "./content/dog-2.jpeg"
raw_image = Image.open(IMAGE_PATH).convert("RGB")
def pil_to_base64(img):
   """Convert PIL image to base64 for BBoxWidget"""
   buffer = io.BytesIO()
   img.save(buffer, format="PNG")
   return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


# Create widget for point selection
widget = BBoxWidget(
   image=pil_to_base64(raw_image),
   classes=["positive", "negative"]
)

# Create UI buttons
segment_button = widgets.Button(
   description='🎯 Segment',
   button_style='success',
   tooltip='Run segmentation with marked points',
   icon='check',
   layout=widgets.Layout(width='150px', height='40px')
)
segment_button.on_click(segment_from_widget)
reset_button = widgets.Button(
   description='🔄 Reset',
   button_style='warning',
   tooltip='Clear all points',
   icon='refresh',
   layout=widgets.Layout(width='150px', height='40px')
)
reset_button.on_click(reset_widget)

# Display UI
print("=" * 70)
print("🎨 INTERACTIVE SAM3 SEGMENTATION WITH BOUNDING BOX WIDGET")
print("=" * 70)
print("\n📋 Instructions:")
print("  1. Draw SMALL boxes on the image where you want to mark points")
print("  2. Label them as 'positive' (object) or 'negative' (background)")
print("  3. The CENTER of each box will be used as a point coordinate")
print("  4. Click 'Segment' button to run SAM3")
print("  5. Click 'Reset' to clear all points and start over")
print("\n💡 Tips:")
print("  • Draw tiny boxes - just big enough to see")
print("  • Positive points = parts of the object you want")
print("  • Negative points = background areas to exclude")
print("\n" + "=" * 70 + "\n")
display(widgets.HBox([segment_button, reset_button]))
display(widget)