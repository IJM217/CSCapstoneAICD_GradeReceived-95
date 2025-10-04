//ContentIngest: pure UI component for text input, word count display, and upload buttons. does NOT manage state or handle file processing.
import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";
import { Clipboard, Upload, FileText, Info, Image as ImageIcon } from "lucide-react";

import { MIN_WORDS, MAX_WORDS } from "./constants";

//main content input component that renders textarea, upload buttons, and word count validation
export default function ContentIngest({
  inputText, //current text content in the textarea
  setInputText, //function to update text content
  textareaRef, //reference to the textarea element for focus control
  fileInputRef, //reference to hidden file input for document uploads
  imageInputRef, //reference to hidden file input for image uploads
  handlePaste, //function to handle clipboard paste operation
  handleFileUpload, //function to handle document file uploads
  handleImageUpload, //function to handle image file uploads
  isCodeMode, //boolean indicating if code analysis mode is enabled
  highlightSegments, //array of text segments with AI detection highlights
  hasHighlights, //boolean indicating if there are any highlights to display
  wordCount, //current word count of the input text
  inRange, //boolean indicating if word count is within valid range
  isLoading, //boolean indicating if analysis is currently running
  uploadedFile, //currently uploaded file object
  handleDetect //function to trigger detection analysis
}) {
  return (
    <>
      <div className="relative">
        {hasHighlights ? (
          //display text with AI detection highlights when analysis results are available
          <div className="min-h-[450px] p-4 scout-textarea rounded-lg leading-6 whitespace-pre-wrap overflow-auto">
            {highlightSegments.map((seg, idx) =>
              seg.ai ? (
                //highlight AI-generated text segments with colored background
                <mark
                  key={idx}
                  className="scout-highlight rounded-sm px-1 py-0.5 font-medium"
                  data-confidence={typeof seg.score === "number" ? seg.score : undefined}
                >
                  {seg.text}
                </mark>
              ) : (
                //display human-written text segments normally
                <span key={idx}>{seg.text}</span>
              )
            )}
          </div>
        ) : (
          //main text input area when no highlights are present
          <Textarea
            ref={textareaRef}
            className="scout-textarea min-h-[450px] max-h-[450px] pr-28 resize-none"
            placeholder={
              isCodeMode
                ? `Enter code for analysis (${MIN_WORDS}–${MAX_WORDS} words)…`
                : `Enter text for analysis (${MIN_WORDS}–${MAX_WORDS} words)…`
            }
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
          />
        )}

        {inputText.length === 0 && !hasHighlights && (
          //action buttons overlay when textarea is empty
          <div className="absolute right-2 top-2 flex gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={handlePaste}
              className="h-9 rounded-full gap-2"
            >
              <Clipboard className="h-4 w-4" />
              Paste Text
            </Button>

            {!isCodeMode && (
              <>
                {/*hidden file input for document upload*/}
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".docx,.pdf,.txt"
                  onChange={(e) => handleFileUpload(e.target.files?.[0])}
                  className="hidden"
                />
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => fileInputRef.current?.click()}
                  className="h-9 rounded-full gap-2"
                >
                  <Upload className="h-4 w-4" />
                  Upload File
                </Button>

                {/*hidden file input for image upload*/}
                <input
                  ref={imageInputRef}
                  type="file"
                  accept="image/*"
                  onChange={(e) => handleImageUpload(e.target.files?.[0])}
                  className="hidden"
                />
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => imageInputRef.current?.click()}
                  className="h-9 rounded-full gap-2"
                >
                  <ImageIcon className="h-4 w-4" />
                  Upload Image
                </Button>
              </>
            )}
          </div>
        )}
      </div>

      <div className="space-y-2">
        {/*word count display*/}
        <div className="text-sm text-foreground/80 font-medium">
          Word count: {wordCount} / {MAX_WORDS}
        </div>
        {wordCount > 0 && wordCount < MIN_WORDS && (
          //warning message when text is too short for accurate analysis
          <div className="flex items-center gap-2 text-sm text-destructive bg-destructive/10 p-3 rounded-lg border border-destructive/20">
            <span className="font-medium">Minimum {MIN_WORDS} words needed</span>
            <div className="relative group">
              <Info className="w-4 h-4 cursor-help text-destructive" />
              <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 px-3 py-2 bg-foreground text-background text-xs rounded-lg opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none whitespace-nowrap z-10">
                Our detection algorithms need at least {MIN_WORDS} words for accurate analysis.
                <div className="absolute top-full left-1/2 transform -translate-x-1/2 w-0 h-0 border-l-4 border-r-4 border-t-4 border-transparent border-t-foreground"></div>
              </div>
            </div>
          </div>
        )}
        {wordCount >= MIN_WORDS && wordCount <= MAX_WORDS && (
          //success message when text length is optimal
          <div className="text-sm text-green-600 bg-green-50 p-3 rounded-lg border border-green-200 font-medium">
            Optimal text length
          </div>
        )}
        {wordCount > MAX_WORDS && (
          //error message when text exceeds maximum allowed length
          <div className="flex items-center gap-2 text-sm text-destructive bg-destructive/10 p-3 rounded-lg border border-destructive/20">
            <span className="font-medium">Maximum {MAX_WORDS} words allowed</span>
          </div>
        )}
      </div>

      {/*main detection button with conditional disabling based on validation rules*/}
      <Button onClick={handleDetect} disabled={
          isLoading ||
          (isCodeMode && inputText.trim().split(/\s+/).length < MIN_WORDS) ||
          (!isCodeMode && uploadedFile && uploadedFile.type?.startsWith("image/") && !uploadedFile) ||
          (!isCodeMode && !uploadedFile && inputText.trim().split(/\s+/).length < MIN_WORDS)
        } className="w-full">        {isLoading ? "Analyzing..." : "Detect"}
      </Button>
    </>
  );
}
