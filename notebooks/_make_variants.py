"""Create a clean variant notebook while preserving the uploaded executed notebook."""
import copy
from pathlib import Path

import nbformat

folder = Path(__file__).resolve().parent
nb = copy.deepcopy(nbformat.read(folder / "prior_pairs_to_scribble_colab.ipynb", as_version=4))
for cell in nb.cells:
    if cell.cell_type == "code":
        cell.outputs = []
        cell.execution_count = None
nb.metadata.pop("widgets", None)
nb.cells[0].source += """

**Bản đa dạng scribble.** Bản notebook có output của lần chạy trước được giữ nguyên.
Mặc định xuất từ SIGMA visual theo đánh giá định tính 8 cặp; text vẫn được chạy để so sánh.
Sáu kiểu nét: box, ellipse, freehand_loop, cross, hatch, short_strokes.
Chỉ dùng cross cho remove; hatch/short_strokes cho remove/change. Add/replace dùng nét khoanh.
"""
nb.cells[2].source = nb.cells[2].source.replace("N_VARIANTS = 3", "N_VARIANTS = 6").replace(
    'EXPORT_METHOD = "sigma_text"', 'EXPORT_METHOD = "sigma_visual"')
nb.cells[2].source += '\nSCRIBBLE_STYLES = ["box", "ellipse", "freehand_loop", "cross", "hatch", "short_strokes"]\nSCRIBBLE_VERSION = "v2-mixed-styles"\n'
nb.cells[14].source = nb.cells[14].source[:nb.cells[14].source.index("def draw_scribble")]
nb.cells[14].source += '''def allowed_styles(operation):
    common = ["box", "ellipse", "freehand_loop"]
    if operation == "remove":
        return common + ["cross", "hatch", "short_strokes"]
    if operation == "change":
        return common + ["hatch", "short_strokes"]
    return common

def style_for(rec, variant):
    choices = [s for s in allowed_styles(rec["operation"]) if s in SCRIBBLE_STYLES]
    if not choices:
        raise ValueError("No suitable scribble styles for operation")
    random.Random(SEED + int(rec["id"][:8], 16)).shuffle(choices)
    return choices[variant % len(choices)]

def draw_scribble(image, boxes, seed, style="box", mask=None):
    if style not in {"box", "ellipse", "freehand_loop", "cross", "hatch", "short_strokes"}:
        raise ValueError(style)
    if style in {"hatch", "short_strokes"} and mask is None:
        raise ValueError("Mask required for interior scribble strokes")
    rng = np.random.default_rng(seed)
    base = Image.fromarray(image).convert("RGBA")
    layer = Image.new("RGBA", base.size)
    draw = ImageDraw.Draw(layer)
    color = [(255, 30, 30), (20, 210, 70), (20, 100, 255)][int(rng.integers(3))]
    rgba = (*color, int(rng.integers(190, 256)))
    for x0, y0, x1, y1 in boxes:
        bw, bh = x1-x0, y1-y0
        cx, cy = (x0+x1-1)/2, (y0+y1-1)/2
        width = max(1, min(round(min(base.size)*rng.uniform(.003, .007)), round(min(bw,bh)*.06)))
        jitter = max(.5, min(bw,bh)*rng.uniform(.01, .04))
        def line(points):
            points = [(round(np.clip(x,0,base.width-1)), round(np.clip(y,0,base.height-1))) for x,y in points]
            draw.line(points, fill=rgba, width=width, joint="curve")
        if style == "box":
            corners = [(x0,y0), (x1-1,y0), (x1-1,y1-1), (x0,y1-1), (x0,y0)]
            points = []
            for start,end in zip(corners,corners[1:]):
                knots = rng.normal(0,jitter,(5,2))
                for t in np.linspace(0,1,24,endpoint=False):
                    delta = [np.interp(t,np.linspace(0,1,5),knots[:,d]) for d in (0,1)]
                    points.append((1-t)*np.array(start)+t*np.array(end)+delta)
            line(points[int(rng.integers(0,6)):])
        elif style in {"ellipse", "freehand_loop"}:
            angle = rng.uniform(-.15,.15)
            phases = rng.uniform(0,2*np.pi,2)
            points = []
            # Enclosing ellipse/loop derives from a coarse box, not the exact result contour.
            for t in np.linspace(0,2*np.pi-rng.uniform(0,.18),120):
                radius = 1 if style == "ellipse" else 1+.06*np.sin(3*t+phases[0])+.035*np.sin(5*t+phases[1])
                x,y = bw*.55*radius*np.cos(t), bh*.55*radius*np.sin(t)
                points.append((cx+x*np.cos(angle)-y*np.sin(angle), cy+x*np.sin(angle)+y*np.cos(angle)))
            line(points)
        elif style == "cross":
            for start,end in (((x0,y0),(x1-1,y1-1)),((x0,y1-1),(x1-1,y0))):
                points = []
                for t in np.linspace(0,1,24):
                    points.append((1-t)*np.array(start)+t*np.array(end)+rng.normal(0,jitter,2))
                line(points)
        elif style == "hatch":
            step = max(width*4, min(bw,bh)*rng.uniform(.15,.25))
            for offset in np.arange(-bh,bw,step):
                line([(x0+offset,y1-1),(x0+offset+bh,y0)])
        else:
            ys,xs = np.where(mask[y0:y1,x0:x1]>0)
            if not len(xs):
                continue
            for _ in range(int(rng.integers(3,7))):
                idx = int(rng.integers(len(xs)))
                center = np.array([x0+xs[idx],y0+ys[idx]])
                angle = rng.uniform(0,2*np.pi)
                direction = np.array([np.cos(angle),np.sin(angle)])
                length = min(bw,bh)*rng.uniform(.2,.5)
                line([center+t*length*direction+rng.normal(0,jitter,2) for t in np.linspace(-.5,.5,16)])
    if style in {"hatch", "short_strokes"}:
        binary = (mask>0).astype(np.uint8)*255
        h,w = binary.shape
        assert (w,h) == base.size
        kernel = max(1, round(min(base.size)*.003))*2+1
        interior = cv2.erode(binary,np.ones((kernel,kernel),np.uint8))
        # Preserve very small components that vanish under erosion.
        count, labels = cv2.connectedComponents(binary)
        for label in range(1,count):
            component = labels == label
            if not np.any(interior[component]):
                interior[component] = 255
        alpha = np.asarray(layer.getchannel("A"),dtype=np.uint8)
        layer.putalpha(Image.fromarray(np.minimum(alpha,interior)))
    return Image.alpha_composite(base,layer).convert("RGB")

# Offline smoke checks for all styles: deterministic, visible, same image dimensions.
_heat = np.zeros((160,240),np.float32)
_heat[30:100,25:95] = 1; _heat[50:140,150:220] = 1
_mask,_boxes,_flags = mask_and_boxes(_heat,.5)
assert len(_boxes)==2 and not _flags
_image = np.zeros((160,240,3),np.uint8)
for _style in ("box","ellipse","freehand_loop","cross","hatch","short_strokes"):
    _a = np.array(draw_scribble(_image,_boxes,42,_style,_mask))
    _b = np.array(draw_scribble(_image,_boxes,42,_style,_mask))
    assert _a.shape==_image.shape and np.any(_a!=_image) and np.array_equal(_a,_b)
    if _style in {"hatch","short_strokes"}:
        assert not np.any(_a[_mask==0])
assert "cross" not in allowed_styles("add")
print("All six scribble styles passed offline checks")
'''
nb.cells[16].source = nb.cells[16].source.replace(
    'save_button = widgets.Button',
    'style_widget = widgets.Dropdown(options=allowed_styles(pairs[0]["operation"]), description="Scribble")\nsave_button = widgets.Button')
nb.cells[16].source = nb.cells[16].source.replace(
    'overlay = draw_scribble(a, boxes, SEED)',
    'overlay = draw_scribble(a, boxes, SEED, style_widget.value, mask)')
nb.cells[16].source = nb.cells[16].source.replace(
    '    review = reviews.get(review_key(rec, method), {})',
    '    style_widget.options = allowed_styles(rec["operation"])\n    review = reviews.get(review_key(rec, method), {})')
nb.cells[16].source = nb.cells[16].source.replace(
    'threshold_widget.observe(show_preview, names="value")',
    'threshold_widget.observe(show_preview, names="value")\nstyle_widget.observe(show_preview, names="value")')
nb.cells[16].source = nb.cells[16].source.replace(
    'method_widget]), threshold_widget,', 'method_widget]), style_widget, threshold_widget,')
nb.cells[18].source = nb.cells[18].source.replace(
    'sigma_rev=SIGMA_REV, checkpoint_revision=checkpoint_revision)',
    'sigma_rev=SIGMA_REV, checkpoint_revision=checkpoint_revision,\n    scribble_version=SCRIBBLE_VERSION, scribble_styles=SCRIBBLE_STYLES)')
nb.cells[18].source = nb.cells[18].source.replace(
    'overlay = draw_scribble(a, boxes, variant_seed)',
    'style = style_for(rec, variant)\n        overlay = draw_scribble(a, boxes, variant_seed, style, mask)')
nb.cells[18].source = nb.cells[18].source.replace(
    'boxes=boxes, seed=variant_seed,', 'boxes=boxes, seed=variant_seed, scribble_style=style, scribble_version=SCRIBBLE_VERSION,')
nb.cells[18].source = nb.cells[18].source.replace(
    'draw_scribble(a, boxes, SEED)', 'draw_scribble(a, boxes, SEED, style_for(rec, 0), mask)')
# Add a deterministic contact sheet before gallery without changing the executed source file.
nb.cells.insert(15,nbformat.v4.new_markdown_cell("""## Xem các kiểu nét trên một mẫu
Ví dụ minh họa dùng mẫu remove để cả sáu kiểu đều có ý nghĩa.
Nét cross dành cho xóa. Các nét trong mask được dùng cho remove/change;
không dùng contour đích chính xác để gợi hình vật thể cần thêm.
"""))
nb.cells.insert(16,nbformat.v4.new_code_cell('''
example = next((p for p in pairs if p["operation"]=="remove"),pairs[0])
heat = predict_pair(example,EXPORT_METHOD)
mask,boxes,flags = mask_and_boxes(heat,MASK_THRESHOLD)
a,b = read_pair(example)
styles = allowed_styles(example["operation"])
fig,axes = plt.subplots(2,3,figsize=(15,9))
for ax in axes.flat:
    ax.axis("off")
for ax,style in zip(axes.flat,styles):
    ax.imshow(draw_scribble(a,boxes,SEED,style,mask)); ax.set_title(style)
plt.tight_layout(); plt.show()
'''))
nb.cells[-1].source += """
- Đánh giá 8 cặp: visual/text tương đương về vùng chính; text có thêm đốm ở áo khoác
  và bảng biểu tình. Mẫu kính thiên văn→cây có mask lan nền; mẫu tượng sai instruction
  ngay từ result (xóa cả dãy). Cần loại/sửa target, đa dạng nét không sửa được nhãn sai.
- Đa dạng scribble bằng kiểu nét và seed; luôn giữ các biến thể cùng cặp trong một split.
"""
nbformat.validate(nb)
nbformat.write(nb,folder / "prior_pairs_to_scribble_variants_colab.ipynb")
