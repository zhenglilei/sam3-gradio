from __future__ import annotations
import argparse,csv,importlib.util,json,sys,time
from collections import defaultdict
from pathlib import Path
from typing import Any
import cv2
import numpy as np
from PIL import Image
REPO_ROOT=Path(__file__).resolve().parents[1]
SCRIPT_DIR=Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0,str(REPO_ROOT))
def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import module")
    mod=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=mod
    spec.loader.exec_module(mod)
    return mod
base=load_module("pvs_bbox_label_base",SCRIPT_DIR/"run_pvs_bbox_label_eval.py")
pcsmod=load_module("pcs_o3_grouped_base",SCRIPT_DIR/"run_pcs_o3_grouped_eval.py")
def mask_iou(a,b):
    a=np.asarray(a).astype(bool); b=np.asarray(b).astype(bool)
    inter=np.logical_and(a,b).sum(dtype=np.float64)
    union=np.logical_or(a,b).sum(dtype=np.float64)
    return 0.0 if union<=0 else float(inter/union)
def save_mask(path,mask):
    path.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray((np.asarray(mask).astype(bool).astype(np.uint8))*255).save(path)
def coco_ann_mask(ann,width,height):
    mask=np.zeros((height,width),dtype=np.uint8)
    seg=ann.get("segmentation")
    if isinstance(seg,list):
        for poly in seg:
            arr=np.asarray(poly,dtype=np.float32).reshape(-1,2)
            if len(arr)>=3:
                cv2.fillPoly(mask,[np.round(arr).astype(np.int32)],1)
    elif isinstance(seg,dict):
        try:
            from pycocotools import mask as mask_utils
            decoded=mask_utils.decode(seg)
            if decoded.ndim==3:
                decoded=decoded.max(axis=2)
            mask=decoded.astype(np.uint8)
        except Exception:
            pass
    return mask.astype(bool)
def labelme_shape_mask(shape,width,height):
    mask=np.zeros((height,width),dtype=np.uint8)
    pts=np.asarray(shape.get("points") or [],dtype=np.float32).reshape(-1,2)
    st=str(shape.get("shape_type") or "polygon").lower()
    if st=="rectangle" and len(pts)>=2:
        x1,y1=pts.min(axis=0); x2,y2=pts.max(axis=0)
        cv2.rectangle(mask,(int(round(x1)),int(round(y1))),(int(round(x2)),int(round(y2))),1,-1)
    elif len(pts)>=3:
        cv2.fillPoly(mask,[np.round(pts).astype(np.int32)],1)
    return mask.astype(bool)
def load_image_size(path):
    with Image.open(path) as im:
        return im.size
def load_coco_ann(o3_root,dataset,split,ann_id):
    p=Path(o3_root)/dataset/"annotations"/("instances_"+split+".json")
    data=json.load(open(p,encoding="utf-8"))
    anns={str(a.get("id")):a for a in data.get("annotations",[])}
    return anns[str(ann_id)]
def gt_mask_for_pvs_item(item,o3_root,t4_root):
    w,h=load_image_size(item["image_path"])
    if item.get("source")=="O3":
        return coco_ann_mask(load_coco_ann(o3_root,item["dataset"],item["split"],item["annotation_id"]),w,h)
    data=json.load(open(Path(t4_root)/Path(item["image_id"]),encoding="utf-8"))
    idx=int(str(item["annotation_id"]).split("shape")[-1])-1
    return labelme_shape_mask(data.get("shapes",[])[idx],w,h)
def gt_mask_for_pcs_ann(data,ann_id,image_path):
    w,h=load_image_size(image_path)
    anns={str(a.get("id")):a for a in data.get("annotations",[])}
    return coco_ann_mask(anns[str(ann_id)],w,h)
def summarize(rows,out_csv,key_fn):
    groups=defaultdict(lambda:{"total":0,"success":0,"iou_sum":0.0,"gt":0,"pred":0,"matched":0,"runs":0,"strict":0})
    for r in rows:
        for key in key_fn(r):
            g=groups[key]
            if "mask_iou" in r:
                g["total"]+=1; g["success"]+=int(r["success"]); g["iou_sum"]+=float(r["mask_iou"])
            else:
                g["runs"]+=1; g["gt"]+=int(r["gt_count"]); g["pred"]+=int(r["pred_count"]); g["matched"]+=int(r["matched_count"]); g["strict"]+=int(r["strict_all_gt_success"])
    sample=rows[0] if rows else {}
    if "mask_iou" in sample:
        fields=["group_a","group_b","total","success","success_rate","mean_iou"]
        with open(out_csv,"w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
            for key,val in sorted(groups.items()):
                total=val["total"]
                w.writerow({"group_a":key[0],"group_b":key[1],"total":total,"success":val["success"],"success_rate":val["success"]/total if total else "","mean_iou":val["iou_sum"]/total if total else ""})
    else:
        fields=["layer","label","prompt_count","runs","gt","pred","matched","recall","precision","strict_all_gt_rate"]
        with open(out_csv,"w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
            for key,val in sorted(groups.items()):
                gt=val["gt"]; pred=val["pred"]; runs=val["runs"]
                w.writerow({"layer":key[0],"label":key[1],"prompt_count":key[2],"runs":runs,"gt":gt,"pred":pred,"matched":val["matched"],"recall":val["matched"]/gt if gt else "","precision":val["matched"]/pred if pred else "","strict_all_gt_rate":val["strict"]/runs if runs else ""})
def run_pvs(args,out_dir):
    pvs_in=Path(args.pvs_dir); out=out_dir/"pvs"; rows=[]
    predictor=base.init_image_predictor(args.device)
    out.mkdir(parents=True,exist_ok=True)
    pred_jsonl=open(out/"pvs_mask_iou95_predictions.jsonl","w",encoding="utf-8")
    for gi,p in enumerate(sorted((pvs_in/"groups").glob("*/prediction.json")),start=1):
        data=json.load(open(p,encoding="utf-8"))
        with Image.open(data["image_path"]) as im:
            image=im.convert("RGB")
        print(f"PVS [{gi}] {data['group_id']} items={len(data.get('items',[]))}",flush=True)
        state=predictor.set_image(image); gdir=out/"groups"/data["group_id"]
        for item in data.get("items",[]):
            pred=base.predict_from_box(predictor,state,item["bbox_xyxy"])
            gt=gt_mask_for_pvs_item(item,args.o3_root,args.t4_root)
            miou=mask_iou(gt,pred["mask"]); ok=miou>=args.threshold
            sample=base._safe_name(item["sample_id"])
            gt_path=gdir/"gt_masks"/(sample+".png"); pred_path=gdir/"pred_masks"/(sample+".png")
            save_mask(gt_path,gt); save_mask(pred_path,pred["mask"])
            row={"source":data.get("source",""),"layer":data.get("layer",""),"dataset":data.get("dataset",""),"label":item.get("label",""),"group_id":data.get("group_id",""),"sample_id":item.get("sample_id",""),"mask_iou":miou,"success":int(ok),"score":pred.get("score",0.0),"gt_mask":str(gt_path.relative_to(out)),"pred_mask":str(pred_path.relative_to(out))}
            rows.append(row); pred_jsonl.write(json.dumps(row,ensure_ascii=False)+"\n")
    pred_jsonl.close()
    fields=["source","layer","dataset","label","group_id","sample_id","mask_iou","success","score","gt_mask","pred_mask"]
    with open(out/"pvs_mask_iou95_items.csv","w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    summarize(rows,out/"pvs_mask_iou95_summary.csv",lambda r:[("ALL","ALL"),(r["source"],"ALL"),(r["source"],r["layer"]),(r["source"],r["label"])])
    return rows
def greedy_match(gt_masks,pred_masks,threshold):
    pairs=[]
    for gi,g in enumerate(gt_masks):
        for pi,p in enumerate(pred_masks):
            pairs.append((mask_iou(g,p),gi,pi))
    pairs.sort(reverse=True)
    used_g=set(); used_p=set(); matched=[]
    for val,gi,pi in pairs:
        if val<threshold:
            break
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi); used_p.add(pi); matched.append((val,gi,pi))
    return matched
def run_pcs(args,out_dir):
    pcs_in=Path(args.pcs_dir); out=out_dir/"pcs"; rows=[]; predictor=base.init_image_predictor(args.device)
    out.mkdir(parents=True,exist_ok=True)
    for ri,p in enumerate(sorted((pcs_in/"groups").glob("*/prediction.json")),start=1):
        data=json.load(open(p,encoding="utf-8"))
        with Image.open(data["image_path"]) as im:
            image=im.convert("RGB")
        print(f"PCS [{ri}] {data['run_id']} prompts={data.get('prompt_count')}",flush=True)
        pcs=pcsmod.run_pcs_from_boxes(predictor,image,data["prompt_boxes_xyxy"],float(data.get("confidence",0.5)))
        ann_json=Path(args.o3_root)/data["dataset"]/"annotations"/("instances_"+data["split"]+".json")
        ann_data=json.load(open(ann_json,encoding="utf-8"))
        gt_masks=[gt_mask_for_pcs_ann(ann_data,aid,data["image_path"]) for aid in data.get("annotation_ids",[])]
        pred_masks=list(pcs.get("masks",[])); matched=greedy_match(gt_masks,pred_masks,args.threshold)
        gdir=out/"groups"/data["run_id"]
        for i,m in enumerate(gt_masks,1): save_mask(gdir/"gt_masks"/(f"gt_{i:04d}.png"),m)
        for i,m in enumerate(pred_masks,1): save_mask(gdir/"pred_masks"/(f"pred_{i:04d}.png"),m)
        gt_count=len(gt_masks); pred_count=len(pred_masks); matched_count=len(matched)
        row={"layer":data.get("layer",""),"dataset":data.get("dataset",""),"label":data.get("label",""),"run_id":data.get("run_id",""),"prompt_count":int(data.get("prompt_count",0)),"gt_count":gt_count,"pred_count":pred_count,"matched_count":matched_count,"recall":matched_count/gt_count if gt_count else 0.0,"precision":matched_count/pred_count if pred_count else 0.0,"strict_all_gt_success":int(gt_count>0 and matched_count==gt_count),"mean_matched_iou":sum(v for v,_,_ in matched)/matched_count if matched_count else 0.0}
        rows.append(row)
        (gdir/"mask_iou_matches.json").write_text(json.dumps({"run_id":data.get("run_id"),"matches":[{"iou":v,"gt_index":gi+1,"pred_index":pi+1} for v,gi,pi in matched]},ensure_ascii=False,indent=2),encoding="utf-8")
    fields=["layer","dataset","label","run_id","prompt_count","gt_count","pred_count","matched_count","recall","precision","strict_all_gt_success","mean_matched_iou"]
    with open(out/"pcs_mask_iou95_runs.csv","w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    summarize(rows,out/"pcs_mask_iou95_summary.csv",lambda r:[("ALL","ALL","ALL"),(r["layer"],"ALL","ALL"),(r["layer"],r["label"],"ALL"),(r["layer"],r["label"],str(r["prompt_count"])),("ALL","ALL",str(r["prompt_count"]))])
    return rows
def parse_args():
    ap=argparse.ArgumentParser()
    ap.add_argument("--pvs-dir",default=".runtime/eval/pvs_bbox_grouped_eval/full_grouped_20260707_095749")
    ap.add_argument("--pcs-dir",default=".runtime/eval/pcs_o3_grouped_eval/full_o3_all_categories_20260707_105126")
    ap.add_argument("--o3-root",default="/data/zhengqiyuan/ADC_contour/datasets/O3_coco")
    ap.add_argument("--t4-root",default="/data/zhengqiyuan/ADC_contour/datasets/T4/labelme_pairs/original_size")
    ap.add_argument("--out-dir",default="")
    ap.add_argument("--threshold",type=float,default=0.95)
    ap.add_argument("--device",default="auto")
    ap.add_argument("--only",choices=["all","pvs","pcs"],default="all")
    return ap.parse_args()
def main():
    args=parse_args(); out=Path(args.out_dir) if args.out_dir else REPO_ROOT/".runtime"/"eval"/("mask_iou95_eval_"+time.strftime("%Y%m%d_%H%M%S"))
    out.mkdir(parents=True,exist_ok=True)
    pvs_rows=[]; pcs_rows=[]
    if args.only in {"all","pvs"}: pvs_rows=run_pvs(args,out)
    if args.only in {"all","pcs"}: pcs_rows=run_pcs(args,out)
    print("OUTPUT_DIR",out)
    if pvs_rows:
        s=sum(int(r["success"]) for r in pvs_rows); print("PVS",s,"/",len(pvs_rows),s/len(pvs_rows) if pvs_rows else 0)
    if pcs_rows:
        gt=sum(int(r["gt_count"]) for r in pcs_rows); mt=sum(int(r["matched_count"]) for r in pcs_rows); pred=sum(int(r["pred_count"]) for r in pcs_rows); print("PCS matched/gt",mt,"/",gt,"recall",mt/gt if gt else 0,"precision",mt/pred if pred else 0)
if __name__=="__main__": main()