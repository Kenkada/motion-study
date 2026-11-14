/**
 * Motion Study: collector for test runs (Google Apps Script, free).
 *
 * The web app POSTs one ZIP per analysis run (all outputs in every format, report, CSV,
 * frames.json and run_info.json with the models and parameters). This script saves it in
 * ONE folder of your Google Drive that the script creates, one sub-folder per run, unzipped.
 *
 * Narrow permissions: the script only asks for "drive.file" (files and folders this script
 * itself created; it cannot see or touch anything else in your Drive) and for calling the
 * Drive API. It never uses DriveApp, which would require access to the whole Drive.
 * Visitors can only send files here; they cannot list or read anything.
 *
 * Setup (once):
 *   1. https://script.google.com -> New project.
 *   2. Project Settings (gear) -> tick "Show appsscript.json manifest file in editor".
 *   3. Editor: replace appsscript.json with collector/appsscript.json, and Code.gs with this file. Save.
 *   4. Deploy -> New deployment -> type "Web app"
 *        Execute as: Me        Who has access: Anyone
 *      -> Deploy -> authorise (the list should show only "See, edit, create and delete only the
 *      specific Google Drive files you use with this app" and "Connect to an external service")
 *      -> copy the Web app URL (ends in /exec).
 *   5. Put that URL in motion_study/static/config.js (COLLECT_URL) and push.
 * After editing this script, use Deploy -> Manage deployments -> edit -> Version: New version,
 * so the URL stays the same.
 */

const FOLDER_NAME = 'Motion Study - test runs';
const KEY = 'motion-study-v1';        // must match COLLECT_KEY in config.js (keeps out random bots)
const MAX_BYTES = 45 * 1024 * 1024;   // Apps Script accepts about 50 MB per request
const API = 'https://www.googleapis.com/drive/v3/files';
const UPLOAD = 'https://www.googleapis.com/upload/drive/v3/files';
const TYPES = {
  png: 'image/png', jpg: 'image/jpeg', webp: 'image/webp', svg: 'image/svg+xml', html: 'text/html',
  csv: 'text/csv', json: 'application/json', zip: 'application/zip',
};

function call_(method, url, payload, contentType) {
  const opts = { method: method, headers: { Authorization: 'Bearer ' + ScriptApp.getOAuthToken() }, muteHttpExceptions: true };
  if (payload !== undefined) { opts.payload = payload; opts.contentType = contentType; }
  const r = UrlFetchApp.fetch(url, opts);
  if (r.getResponseCode() >= 300) throw new Error(method + ' ' + r.getResponseCode() + ' ' + r.getContentText().slice(0, 300));
  return JSON.parse(r.getContentText() || '{}');
}

function makeFolder_(name, parentId) {
  const meta = { name: name, mimeType: 'application/vnd.google-apps.folder' };
  if (parentId) meta.parents = [parentId];
  return call_('post', API + '?fields=id', JSON.stringify(meta), 'application/json').id;
}

function rootFolder_() {
  const props = PropertiesService.getScriptProperties();
  const id = props.getProperty('folderId');
  if (id) {
    try {
      const f = call_('get', API + '/' + id + '?fields=id,trashed');
      if (!f.trashed) return id;
    } catch (e) { /* deleted: make a new one */ }
  }
  const nid = makeFolder_(FOLDER_NAME);
  props.setProperty('folderId', nid);
  return nid;
}

function saveFile_(blob, name, parentId) {
  const ext = name.split('.').pop().toLowerCase();
  const type = TYPES[ext] || 'application/octet-stream';
  // 1) create the file entry, 2) upload its content
  const id = call_('post', API + '?fields=id', JSON.stringify({ name: name, parents: [parentId], mimeType: type }), 'application/json').id;
  call_('patch', UPLOAD + '/' + id + '?uploadType=media&fields=id', blob.getBytes(), type);
}

function reply_(obj) {
  return ContentService.createTextOutput(JSON.stringify(obj)).setMimeType(ContentService.MimeType.JSON);
}

function clean_(s) {
  return String(s || '').replace(/[^A-Za-z0-9._ -]+/g, '_').slice(0, 80);
}

function doGet() {
  return reply_({ ok: true, app: 'motion-study-collector' });
}

function doPost(e) {
  try {
    const body = JSON.parse(e.postData.contents);
    if (body.app !== 'motion-study' || body.key !== KEY) return reply_({ ok: false, error: 'rejected' });
    const bytes = Utilities.base64Decode(body.zip || '');
    if (!bytes.length || bytes.length > MAX_BYTES) return reply_({ ok: false, error: 'size' });

    const stamp = Utilities.formatDate(new Date(), 'Asia/Kolkata', 'yyyy-MM-dd_HH-mm-ss');
    const name = [stamp, clean_(body.kind), clean_(body.name)].filter(String).join('_');
    const folder = makeFolder_(name, rootFolder_());

    const zip = Utilities.newBlob(bytes, 'application/zip', name + '.zip');
    saveFile_(zip, name + '.zip', folder);
    // also unzipped, so the files open directly in Drive
    Utilities.unzip(zip).forEach(function (b) { saveFile_(b, b.getName().split('/').pop(), folder); });
    return reply_({ ok: true, folder: name });
  } catch (err) {
    return reply_({ ok: false, error: String(err) });
  }
}

// Run this once from the editor (Run -> testSetup) to authorise and create the folder.
function testSetup() {
  Logger.log('Folder id: ' + rootFolder_());
}
