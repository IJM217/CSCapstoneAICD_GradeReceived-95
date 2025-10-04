//page.js: main homepage component that orchestrates text/image detection and displays results
"use client";

import { useState } from "react";
import DetectionPanel from "@/components/detect/DetectionPanel";
import ResultsPanel from "@/components/detect/ResultsPanel";
import { Eye } from "lucide-react";
import { Switch } from "@/components/ui/switch";

//main homepage component that manages detection state and API calls
export default function HomePage() {
  const [detectionResult, setDetectionResult] = useState(null); //current detection results
  const [isLoading, setIsLoading] = useState(false); //loading state for API calls
  const [isCodeMode, setIsCodeMode] = useState(false); //toggle between text and code analysis

  //handles text detection API call for both regular text and code analysis
  const handleDetect = async (text, codeMode = false) => {
    setIsLoading(true);
    try {
      const res = await fetch("/api/submit-text", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, isCodeMode: codeMode })
      });
      if (!res.ok) {
        throw new Error(await res.text());
      }
      const data = await res.json();
      setDetectionResult({
        text,
        verdict: data.verdict,
        score: data.score,
        modelVersion: data.modelVersion,
        highlights: data.highlights || []
      });
    } catch (e) {
      setDetectionResult({ text, verdict: "Service Unavailable", score: 0, error: String(e) });
    } finally {
      setIsLoading(false);
    }
  };

  //handles image detection API call for uploaded images
  const handleImageDetect = async (file) => {
    if (!file) return;
    setIsLoading(true);
    try {
      const formData = new FormData();
      formData.append("image", file);
      const res = await fetch("/api/submit-image", {
        method: "POST",
        body: formData
      });
      if (!res.ok) {
        throw new Error(await res.text());
      }
      const data = await res.json();
      
      setDetectionResult({
        verdict: data.verdict,
        modelVersion: data.modelVersion,
      });
    } catch (e) {
      setDetectionResult({ verdict: "Service Unavailable"});
    } finally {
      setIsLoading(false);
    }
  };

  //clears the current detection results
  const handleClearResult = () => {
    setDetectionResult(null);
  };

  return (
    <div className="mx-auto max-w-7xl px-4 py-8">
      <div className="grid gap-8 lg:grid-cols-3">
        {/*main detection panel - takes up 2/3 of the width on large screens*/}
        <div className="lg:col-span-2">
          <div className="scout-card p-6">
            <div className="mb-6 flex items-center justify-between">
              <h2 className="text-xl font-bold text-foreground mb-3">
                {isCodeMode ? "Code Analysis" : "Text/Image Analysis"}
              </h2>
              {/*mode toggle switch for switching between text and code analysis*/}
              <div className="flex items-center space-x-2">
                <span className="text-l font-bold text-foreground">
                  {isCodeMode ? "Code Detection On" : "Code Detection Off"}
                </span>
                <Switch checked={isCodeMode} onCheckedChange={setIsCodeMode} />
              </div>
            </div>
            {/*instructional text shown when no results are present*/}
            {!detectionResult && (
              <p className="text-foreground/80 text-sm leading-relaxed">
                {isCodeMode ? "Paste code to analyze for AI-generated content." : "Insert text or upload a file to analyse for AI-generated text. Or upload an image to analyse for AI-generated images."}
              </p>
            )}
            {/*main detection component that handles input and analysis*/}
            <DetectionPanel 
              onDetect={handleDetect} 
              onDetectImage={handleImageDetect}
              detectionResult={detectionResult}
              isLoading={isLoading}
              onClearResult={handleClearResult}
              isCodeMode={isCodeMode}
              setIsCodeMode={setIsCodeMode}
            />
          </div>
        </div>
        {/*results panel - takes up 1/3 of the width on large screens*/}
        <div>
          <div className="scout-card p-6">
            <div className="mb-6">
              <h2 className="text-xl font-bold text-foreground mb-3">Analysis Results</h2>
              <p className="text-foreground/80 text-sm leading-relaxed">
                
              </p>
            </div>
            {/*component that displays detection results and scores*/}
            <ResultsPanel detectionResult={detectionResult} isLoading={isLoading} />
          </div>
        </div>
      </div>

      {/*help icon - fixed position in bottom right corner*/}
      <div className="fixed bottom-6 right-6">
        <button
          onClick={() => window.open('/scoutaihelper.pdf', '_blank')}
          className="p-3 bg-primary/10 hover:bg-primary/20 text-primary border border-primary/20 rounded-full transition-colors duration-200 shadow-lg"
          title="How to use SCOUT AI"
        >
          <Eye className="w-5 h-5" />
        </button>
      </div>
    </div>
  );
}
