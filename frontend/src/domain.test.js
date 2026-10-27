import test from 'node:test';
import assert from 'node:assert/strict';
import {validateFile, validateResult, filterFindings, escapeHTML, isDicom, imageMime} from './domain.js';
test('unsupported and oversized files are rejected', () => {
  assert.ok(validateFile({name:'scan.tiff',type:'image/tiff',size:1024}));
  assert.ok(validateFile({name:'scan.png',type:'image/png',size:21*1024*1024}));
  assert.equal(validateFile({name:'scan.jpg',type:'image/jpeg',size:1024}),null);
});
test('all 18 independent scores are required and remain unchanged', () => {
  const result={task:'chest_xray_findings',findings:Array.from({length:18},(_,i)=>({label:`finding ${i}`,score:.8,reference_threshold:.1,above_reference_threshold:true}))};
  assert.equal(validateResult(result),result);
  assert.equal(result.findings.reduce((sum,f)=>sum+f.score,0)>1,true);
  result.findings[1].label=result.findings[0].label;
  assert.throws(()=>validateResult(result));
});
test('filtering does not alter reference flags or source results', () => {
  const findings=[{label:'Pleural_Thickening',above_reference_threshold:true},{label:'Effusion',above_reference_threshold:false}];
  assert.equal(filterFindings(findings,'pleural thick',true).length,1);
  assert.equal(filterFindings(findings,'',true).length,1);
  assert.equal(findings.length,2);
});
test('filenames and error text cannot inject markup',()=>assert.equal(escapeHTML('<img src=x onerror="bad">'),'&lt;img src=x onerror=&quot;bad&quot;&gt;'));

test('DICOM files are identified without claiming browser pixel decoding',()=>{
  const file={name:'study.DCM',type:'',size:2048};
  assert.equal(validateFile(file),null);
  assert.equal(isDicom(file),true);
  assert.equal(imageMime(file),'application/dicom');
  assert.equal(imageMime({name:'scan.png',type:''}),'image/png');
});
