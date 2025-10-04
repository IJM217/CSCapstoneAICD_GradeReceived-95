//fileProcessor: pure utility functions for file processing (PDF, DOCX, TXT, images). does NOT manage React state or UI components.
import * as mammoth from "mammoth";

//dynamically loads PDF.js from a CDN in the browser and sets up the worker
async function loadPdfJsFromCdn() {
  //PDF spacing heuristic: assumes 5px vertical jump = new line, 2px horizontal gap = word boundary
  if (typeof window === "undefined") throw new Error("Not in browser");
  if (window.pdfjsLib) return window.pdfjsLib; //return cached instance if already loaded

  const VERSION = "2.16.105"; //PDF.js version to load from CDN
  const scriptUrl = `https://unpkg.com/pdfjs-dist@${VERSION}/build/pdf.min.js`;
  const workerUrl = `https://unpkg.com/pdfjs-dist@${VERSION}/build/pdf.worker.min.js`;

  //load PDF.js script dynamically
  await new Promise((resolve, reject) => {
    const s = document.createElement("script");
    s.src = scriptUrl;
    s.onload = resolve;
    s.onerror = () => reject(new Error("Failed to load pdf.js from CDN"));
    document.head.appendChild(s);
  });

  //configure worker for PDF processing
  try {
    window.pdfjsLib.GlobalWorkerOptions.workerSrc = workerUrl;
  } catch (err) {
    console.warn("pdfjs worker assignment failed", err);
  }

  return window.pdfjsLib;
}

//handles document file uploads (DOCX, PDF, TXT) and extracts text content
export const handleFileUpload = async (file, setUploadedFile, setText, textareaRef) => {
  if (!file) return;
  setUploadedFile(file); //update uploaded file state

  const fileExt = file.name.split(".").pop()?.toLowerCase(); //get file extension

  if (fileExt === "docx") {
    //process DOCX files using mammoth library
    const reader = new FileReader();
    reader.onload = async (event) => {
      try {
        const arrayBuffer = event.target.result;
        const result = await mammoth.extractRawText({ arrayBuffer });
        setText(result.value);
        textareaRef.current?.focus();
      } catch (error) {
        console.error("Failed to extract DOCX text:", error);
        alert("Failed to extract text from DOCX file. Please try a different file.");
      }
    };
    reader.readAsArrayBuffer(file);
 } else if (fileExt === "pdf") {
  try {
    const pdfjsLib = await loadPdfJsFromCdn(); //load PDF.js library

    const typedArray = new Uint8Array(await file.arrayBuffer());
    const pdf = await pdfjsLib.getDocument({ data: typedArray }).promise;

    let extractedText = "";
    //process each page of the PDF
    for (let i = 1; i <= pdf.numPages; i++) {
 const page = await pdf.getPage(i);
 const content = await page.getTextContent();

 let pageText = "";
 let lastY = null; //track vertical position for line breaks

 content.items.forEach((item, idx) => {
   const tx = item.transform;
   const x = tx[4]; //horizontal position
   const y = tx[5]; //vertical position

   //insert a newline if we jumped vertically (new line of text)
   if (lastY !== null && Math.abs(y - lastY) > 5) {
     pageText += "\n";
   }

   //add space if this isn't the very start and the previous char didn't end with space
   if (
     idx > 0 &&
     !pageText.endsWith(" ") &&
     !item.str.startsWith(" ") &&
     x - (content.items[idx - 1].transform[4] + content.items[idx - 1].width) > 2
   ) {
     pageText += " ";
   }

   pageText += item.str;
   lastY = y;
 });

 extractedText += pageText + "\n\n";
}

   setText(extractedText || "No text found in PDF.");
   textareaRef.current?.focus();
 } catch (err) {
   console.error("PDF extraction failed (CDN)", err);
   alert("Could not extract PDF text in the browser. Try again or use the server-side option.");
 }
} else if (fileExt === "txt") {
   //process plain text files
   const reader = new FileReader();
   reader.onload = (event) => {
     setText(event.target.result || "Empty text file.");
     textareaRef.current?.focus();
   };
   reader.readAsText(file);

 } else {
   alert("Unsupported file type. Please upload .docx, .pdf, or .txt.");
 }
};

//handles image file uploads and triggers image detection analysis
export const handleImageUpload = (file, setUploadedFile, onDetectImage, imageInputRef) => {
  if (!file) return;
  setUploadedFile(file); //update uploaded file state
  if (imageInputRef.current) imageInputRef.current.value = ""; //clear input for re-upload
};