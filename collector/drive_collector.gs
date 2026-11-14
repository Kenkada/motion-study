/**
 * Motion Study: collector for test runs (Google Apps Script, free).
 *
 * The web app POSTs one ZIP per analysis run (all outputs in every format, report, CSV,
 * frames.json and run_info.json with the models and parameters). This script saves it in
 * a private folder of YOUR Google Drive, one sub-folder per run, unzipped.
 * Visitors can only send files here; they cannot list or read anything in the folder.
 *
 * Setup (once):
 *   1. https://script.google.com -> New project -> paste this file -> Save.
 *   2. Deploy -> New deployment -> type "Web app"
 *        Execute as: Me        Who has access: Anyone
 *      -> Deploy -> authorise -> copy the Web app URL (ends in /exec).
 *   3. Put that URL in motion_study/static/config.js (COLLECT_URL) and push.
 * After editing this script, use Deploy -> Manage deployments -> edit -> Version: New version,
 * so the URL stays the same.
 */

const FOLDER_NAME = 'Motion Study - test runs';
const KEY = 'motion-study-v1';        // must match COLLECT_KEY in config.js (keeps out random bots)
const MAX_BYTES = 45 * 1024 * 1024;   // Apps Script accepts about 50 MB per request

function rootFolder_() {
  const props = PropertiesService.getScriptProperties();
  const id = props.getProperty('folderId');
  if (id) { try { return DriveApp.getFolderById(id); } catch (e) { /* deleted: make a new one */ } }
  const f = DriveApp.createFolder(FOLDER_NAME);
  props.setProperty('folderId', f.getId());
  return f;
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
    const folder = rootFolder_().createFolder(name);

    const zip = Utilities.newBlob(bytes, 'application/zip', name + '.zip');
    folder.createFile(zip);
    // also unzipped, so the files open directly in Drive
    Utilities.unzip(zip).forEach(function (b) { folder.createFile(b.setName(b.getName().split('/').pop())); });
    return reply_({ ok: true, folder: name });
  } catch (err) {
    return reply_({ ok: false, error: String(err) });
  }
}
