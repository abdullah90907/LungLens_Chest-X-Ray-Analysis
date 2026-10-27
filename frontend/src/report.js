import {escapeHTML as esc, displayLabel} from './domain.js';

export function downloadBlob(content, name, type) {
  const url = URL.createObjectURL(new Blob([content],{type}));
  const link = document.createElement('a'); link.href=url; link.download=name;
  document.body.append(link); link.click(); link.remove();
  setTimeout(()=>URL.revokeObjectURL(url),1500);
}

async function embeddedImage(source) {
  if(!source) return null;
  if(/^data:image\/png;base64,[A-Za-z0-9+/=]+$/.test(source)) return source;
  if(!source.startsWith('blob:')) return null;
  const image=new Image(); image.src=source; await image.decode();
  const scale=Math.min(1,1024/Math.max(image.width,image.height));
  const canvas=document.createElement('canvas');canvas.width=Math.round(image.width*scale);canvas.height=Math.round(image.height*scale);
  canvas.getContext('2d').drawImage(image,0,0,canvas.width,canvas.height);
  return canvas.toDataURL('image/png');
}

export async function downloadResearchReport(result, {includeImages=false, source=null, heatmaps=[], anatomy=null}={}) {
  const sections=[];
  if(includeImages) {
    const input=await embeddedImage(source);
    if(input)sections.push(`<figure><h3>Input preview</h3><img src="${input}" alt="Input preview"/><figcaption>Pixels may contain identifying text. This report is not certified de-identified.</figcaption></figure>`);
    for(const map of heatmaps)if(/^[A-Za-z0-9+/=]+$/.test(map.overlay_png_base64))sections.push(`<figure><h3>${esc(displayLabel(map.finding))} · Grad-CAM</h3><img src="data:image/png;base64,${map.overlay_png_base64}" alt="Finding attribution overlay"/><figcaption>${esc(map.notice)} Original, unfiltered model rendering; interactive viewer display adjustments are not included. Crop: ${esc(JSON.stringify(map.crop_box))}.</figcaption></figure>`);
    if(anatomy && /^[A-Za-z0-9+/=]+$/.test(anatomy.overlay_png_base64))sections.push(`<figure><h3>Anatomical segmentation</h3><img src="data:image/png;base64,${anatomy.overlay_png_base64}" alt="Anatomical overlay"/><figcaption>${esc(anatomy.notice)} These are all three generated anatomy layers, independent of viewer visibility toggles.</figcaption></figure>`);
  }
  const quality=result.quality;
  const metadata=result.input_metadata || {};
  const html=`<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>LungLens research report</title><style>
  *{box-sizing:border-box}body{font:14px/1.7 Arial,sans-serif;color:#203d35;background:#f1f6f2;margin:0;padding:35px}main{max-width:950px;margin:auto;background:#fff;padding:40px;border:1px solid #dce7dd;border-radius:12px}h1{font-size:32px;letter-spacing:-1px;margin:5px 0}h2{font-size:20px;margin-top:30px;border-bottom:1px solid #dce7dd;padding-bottom:8px}h3{font-size:15px}.eyebrow{color:#4b8467;font-size:11px;letter-spacing:2px}.notice{padding:14px 17px;background:#f0f6ef;border-left:3px solid #42896a;margin:20px 0}dl{display:grid;grid-template-columns:160px 1fr;gap:7px 15px}dt{color:#718477}dd{margin:0;overflow-wrap:anywhere}table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;padding:10px 8px;border-bottom:1px solid #e5ece2}th{background:#f5f8f1}td:not(:first-child){font-variant-numeric:tabular-nums}figure{margin:25px 0;break-inside:avoid}figure img{max-width:100%;max-height:550px;display:block;margin:auto;background:#132b2b;border-radius:8px}figcaption{font-size:12px;color:#718477;margin-top:10px}.footer{margin-top:35px;padding-top:15px;border-top:1px solid #dce7dd;font-size:12px;color:#718477}@media print{body{background:white;padding:0}main{border:0;padding:10mm;max-width:none}tr{break-inside:avoid}.notice{print-color-adjust:exact}a{color:inherit}}
  </style></head><body><main><span class="eyebrow">LUNGLENS / RESEARCH OUTPUT</span><h1>Chest X-ray research report</h1><p>Exploratory model output. Not a clinical radiology report.</p><div class="notice">${esc(result.notice)} Heatmaps are model attributions, not confirmed disease locations. Anatomical masks are not pathology segmentation.</div><h2>Analysis details</h2><dl><dt>Analysis ID</dt><dd>${esc(result.analysis_id)}</dd><dt>Created</dt><dd>${esc(result.created_at)}</dd><dt>Declared view</dt><dd>${esc(result.image.view || 'Not provided')} · image modality not automatically verified</dd><dt>Input dimensions</dt><dd>${result.image.width} × ${result.image.height}</dd><dt>Source format</dt><dd>${esc(metadata.format || 'Image')}</dd><dt>Windowing</dt><dd>${esc(metadata.windowing || 'Source export')}</dd><dt>Model weights</dt><dd>${esc(result.model.weights)}</dd><dt>Library version</dt><dd>${esc(result.model.library_version)}</dd><dt>Checkpoint SHA-256</dt><dd>${esc(result.model.checkpoint_sha256)}</dd></dl><h2>Independent finding scores</h2><p>Scores are uncalibrated and do not sum to 1. Published reference cutoffs have not been locally clinically validated.</p><table><thead><tr><th>Finding</th><th>Model score</th><th>Reference</th><th>Comparison</th></tr></thead><tbody>${result.findings.map(f=>`<tr><td>${esc(displayLabel(f.label))}</td><td>${f.score.toFixed(4)}</td><td>${f.reference_threshold.toFixed(4)}</td><td>${f.above_reference_threshold?'Above reference':'Below reference'}</td></tr>`).join('')}</tbody></table><h2>Input inspection</h2><p>${esc(quality?.notice || 'Quality inspection was not available.')}</p>${quality?.flags?.length?`<ul>${quality.flags.map(f=>`<li>${esc(f.message)}</li>`).join('')}</ul>`:'<p>No heuristic flags were returned. This does not establish clinical image quality.</p>'}<h2>Limitations</h2><ul>${(result.limitations || []).map(text=>`<li>${esc(text)}</li>`).join('')}</ul><p>Independent accuracy evaluation has not been completed for this application. No sensitivity, specificity or accuracy claims are made by this report.</p>${sections.length?`<h2>Optional research visualizations</h2>${sections.join('')}`:'<h2>Visualizations</h2><p>Images were excluded from this report. To include previews and generated overlays, explicitly select the image-inclusion option before downloading.</p>'}<div class="footer">LungLens · Built for the Build with AI Hackathon by Gen AI.<br>This document is a self-contained HTML export. Use your browser’s Print command to save a PDF. Patient identifiers from DICOM metadata are not included; embedded pixels may still contain identifiers.</div></main></body></html>`;
  downloadBlob(html,`lunglens-research-${String(result.analysis_id).replace(/[^\w-]/g,'')}.html`,'text/html;charset=utf-8');
}
