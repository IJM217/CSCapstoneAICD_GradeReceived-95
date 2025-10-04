//DetectionPanel: main orchestrator component that manages state, validation, and coordinates between ContentIngest and fileProcessor. does NOT handle file processing or UI rendering directly.
"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { FileText, X } from "lucide-react";
import { useDetectionHighlights } from "./useDetectionHighlights";
import ContentIngest from "./ContentIngest";
import { handleFileUpload, handleImageUpload } from "./fileProcessor";
import { MIN_WORDS, MAX_WORDS } from "./constants";

//main detection panel component that orchestrates text/image analysis workflow
export default function DetectionPanel({
  onDetect, //function to trigger text detection analysis
  onDetectImage, //function to trigger image detection analysis
  detectionResult, //current detection results from analysis
  isLoading, //boolean indicating if analysis is currently running
  onClearResult, //function to clear detection results
  isCodeMode //boolean indicating if code analysis mode is enabled
}) {
  //state management for input text and file handling
  const [inputText, setInputText] = useState(""); //current text content for analysis
  const [uploadedFile, setUploadedFile] = useState(null); //currently uploaded file
  const [lastAnalyzedText, setLastAnalyzedText] = useState(""); //text that was last analyzed for highlighting

  //refs for DOM element access
  const textareaRef = useRef(null); //reference to main textarea element
  const fileInputRef = useRef(null); //reference to hidden file input for documents
  const imageInputRef = useRef(null); //reference to hidden file input for images

  //computed values for validation
  const wordCount = inputText.trim() ? inputText.trim().split(/\s+/).length : 0; //current word count
  const inRange = wordCount >= MIN_WORDS && wordCount <= MAX_WORDS; //whether word count is valid

  //handles clipboard paste operation and updates input text
  const handlePaste = useCallback(async () => {
    try {
      const clipboardText = await navigator.clipboard.readText();
      setInputText(clipboardText);
      textareaRef.current?.focus();
    } catch (error) {
      console.warn("Failed to access clipboard:", error);
    }
  }, []);

  //main detection handler that determines whether to analyze text or image based on current state
  const handleDetect = useCallback(() => {
    if (isCodeMode) {
      //code analysis mode - analyze text as code
      if (inputText.trim().length >= MIN_WORDS) {
        setLastAnalyzedText(inputText);
        onDetect(inputText, true);
      }
    } else {
      if (uploadedFile) {
        //if it's an image, call image detection
        if (uploadedFile.type?.startsWith("image/")) {
          onDetectImage(uploadedFile);
        } else {
          //otherwise, call text detection
          onDetect(inputText, false);
        }
      } else if (inputText.trim().length >= MIN_WORDS) {
        //text analysis mode - analyze text as regular content
        setLastAnalyzedText(inputText);
        onDetect(inputText, false);
      }
    }
  }, [inputText, uploadedFile, isCodeMode, onDetect, onDetectImage]);
  
  //clears all input state and resets the form to initial state
  const handleClear = useCallback(() => {
    setInputText("");
    setLastAnalyzedText("");
    if (onClearResult) onClearResult();
    textareaRef.current?.focus();
    if (fileInputRef.current) fileInputRef.current.value = "";
    if (imageInputRef.current) imageInputRef.current.value = "";
    setUploadedFile(null);
  }, [onClearResult]);

  //get highlight segments for displaying AI detection results in text
  const highlightSegments = useDetectionHighlights(inputText, detectionResult, lastAnalyzedText);
  const hasHighlights = highlightSegments.length > 0; //whether there are highlights to display

  return (
    <div className="space-y-3">
      {uploadedFile && (
        //file upload indicator with remove button
        <div className="mb-4 p-3 bg-primary/10 border-2 border-primary/30 rounded-lg flex items-center justify-between">
          <div className="flex items-center gap-2">
            <FileText className="w-4 h-4 text-primary" /> 
            <span className="text-sm font-medium text-foreground">{uploadedFile.name}</span>
          </div>
          <Button
            type="button"
            variant="ghost"
            size="sm"
            onClick={handleClear}
            className="h-6 w-6 p-0 hover:bg-primary/20 text-primary"
            aria-label="Remove file"
          >
            <X className="w-4 h-4" />
          </Button>
        </div>
      )}

      {/*main content input component*/}
      <ContentIngest
        inputText={inputText}
        setInputText={setInputText}
        textareaRef={textareaRef}
        fileInputRef={fileInputRef}
        imageInputRef={imageInputRef}
        handlePaste={handlePaste}
        uploadedFile={uploadedFile}
        handleFileUpload={(file) => handleFileUpload(file, setUploadedFile, setInputText, textareaRef)}
        handleImageUpload={(file) => handleImageUpload(file, setUploadedFile, onDetectImage, imageInputRef)}
        isCodeMode={isCodeMode}
        highlightSegments={highlightSegments}
        hasHighlights={hasHighlights}
        wordCount={wordCount}
        inRange={inRange}
        isLoading={isLoading}
        handleDetect={handleDetect}
      />

      {/*clear/new analysis button*/}
      <Button onClick={handleClear} variant="outline" className="w-full">
        {hasHighlights ? "New Analysis" : "Clear"}
      </Button>
    </div>
  );
}