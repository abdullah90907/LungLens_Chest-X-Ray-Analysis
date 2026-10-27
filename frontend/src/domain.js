export const MAX_BYTES = 20 * 1024 * 1024;
export const escapeHTML = (value) => String(value ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
export const displayLabel = label => label.replaceAll('_', ' ');
export const isDicom = file => /\.(dcm|dicom)$/i.test(file?.name || '') || file?.type === 'application/dicom';
export const imageMime = file => isDicom(file) ? 'application/dicom' : (file.type || (/\.png$/i.test(file.name) ? 'image/png' : 'image/jpeg'));
export function validateFile(file) {
  if (!file) return 'Choose an image to continue.';
  if (file.size > MAX_BYTES) return 'This image is larger than 20 MB. Choose a smaller export.';
  if (isDicom(file)) return null;
  if (!/\.(png|jpe?g)$/i.test(file.name) || (file.type && !['image/png','image/jpeg'].includes(file.type))) return 'Choose an 8-bit PNG/JPEG or a supported CR/DX DICOM chest X-ray.';
  return null;
}
export function validateResult(result) {
  if (result?.task !== 'chest_xray_findings' || !Array.isArray(result.findings) || result.findings.length !== 18) throw new Error('The backend returned an incomplete analysis. Please try again.');
  const labels = new Set();
  for (const finding of result.findings) {
    if (typeof finding.label !== 'string' || labels.has(finding.label) || !Number.isFinite(finding.score) || finding.score < 0 || finding.score > 1 || !Number.isFinite(finding.reference_threshold) || finding.reference_threshold < 0 || finding.reference_threshold > 1 || typeof finding.above_reference_threshold !== 'boolean') throw new Error('The backend returned invalid finding scores. Please try again.');
    labels.add(finding.label);
  }
  return result;
}
export function filterFindings(findings, query, flaggedOnly) {
  return findings.filter(f => (!flaggedOnly || f.above_reference_threshold) && displayLabel(f.label).toLowerCase().includes(query.toLowerCase()));
}
export function apiError(body, status) {
  if (typeof body?.detail === 'string') return body.detail;
  if (status === 413) return 'The upload exceeds the 20 MB limit.';
  if (status === 422) return 'This image could not be analyzed. Use a frontal AP/PA chest X-ray as an 8-bit PNG/JPEG or supported CR/DX DICOM file.';
  if (status === 503) return 'The analysis engine is unavailable. Check that the pretrained weights are installed, then try again.';
  return 'The request could not be completed. Please try again.';
}
