//ResultsPanel: displays detection results with verdict, score, and copy functionality. does NOT perform detection or manage detection state.
"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";

//component that displays AI detection results including score, verdict, and export options
export default function ResultsPanel({ detectionResult, isLoading }) {
  const [copied, setCopied] = useState(false); //state for copy button feedback

  //show loading state while analysis is running
  if (isLoading) return <div className="text-sm text-gray-600" aria-live="polite">Analyzing…</div>;
  
  //show empty state when no results are available
  if (!detectionResult) return (
    <div className="space-y-6">
      <div className="text-center py-12">
        <div className="w-16 h-16 bg-primary/10 rounded-full flex items-center justify-center mx-auto mb-4">
          <div className="text-2xl">🔍</div>
        </div>
        <div className="text-sm text-foreground/80 mb-2 font-medium">Analysis results will appear here</div>
        <div className="text-xs text-foreground/60">Run an analysis to see detailed results</div>
      </div>
      
      {/*legend showing what the colored indicators represent*/}
      <div className="space-y-3 text-sm">
        <div className="font-semibold text-foreground mb-3">Content Classification:</div>
        <div className="flex items-center justify-between p-3 bg-destructive/10 rounded-lg border border-destructive/20">
          <span className="font-medium">AI-Generated</span>
          <span className="scout-highlight h-3 w-3 rounded-sm" />
        </div>
        <div className="flex items-center justify-between p-3 bg-muted/50 rounded-lg border border-border">
          <span className="font-medium">Human-Generated</span>
          <span className="inline-block h-3 w-3 rounded-sm bg-muted border border-border" />
        </div>
      </div>
    </div>
  );

  //copies the analysis summary to clipboard for easy sharing
  const handleCopySummary = async () => {
    const summary = `AI Content Detector — Verdict: ${detectionResult.verdict}, Score: ${
      typeof detectionResult.score === "number" ? detectionResult.score : "N/A"
    }${detectionResult.modelVersion ? `, Model: ${detectionResult.modelVersion}` : ""}`;
    try {
      await navigator.clipboard.writeText(summary);
      setCopied(true);
     //setTimeout(() => setCopied(false), 1400000000);
    } catch (error) {
      console.warn("Clipboard copy failed:", error);
    }
  };

  //downloads the complete analysis results as a JSON file
  const handleDownloadJson = () => {
    const payload = {
      verdict: detectionResult.verdict,
      modelVersion: detectionResult.modelVersion ?? "desklib/ai-text-detector-v1.01",
      highlights: detectionResult.highlights ?? [],
      score: detectionResult.score,
      text: detectionResult.text ?? "",
      createdAt: detectionResult.createdAt ?? new Date().toISOString(),
      
    };
    
    //creating a blob and download link
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const downloadLink = document.createElement("a");
    downloadLink.href = url;
    downloadLink.download = "ai-detection-result.json";
    downloadLink.click();
    URL.revokeObjectURL(url);
  };

  //determine indicator color based on AI detection result
  const isLikelyAI = detectionResult.verdict === "LIKELY_AI";
  const dotColor = isLikelyAI ? "scout-highlight" : "bg-muted border border-border";

  return (
    <div className="space-y-6" aria-live="polite">
      {/*main percentage result display - only show if score is available*/}
      {typeof detectionResult.score === "number" && (
        <div className="text-center">
          <div className="text-6xl font-bold text-gray-900">
            {detectionResult.score}%
          </div>
          <div className="text-sm text-gray-600 mt-2">
            probability of AI generation
          </div>
        </div>
      )}

      {/*verdict display with colored indicator*/}
      <div className="text-sm">
        <div className="flex items-center gap-2">
          <span className="font-bold text-2xl">Verdict:</span> 
          <span className="font-bold text-2xl">{detectionResult.verdict}</span>
          <span className={`inline-block h-3 w-3 rounded-sm ${dotColor} ring-1 ring-inset ring-gray-300`} />
        </div>
      </div>

      {/*action buttons for copying and downloading results*/}
      <div className="flex items-center gap-2 pt-1">
        <Button
          variant="outline"
          size="sm"
          onClick={handleCopySummary}
          className="h-8 rounded-full bg-white px-3 border-gray-200 text-gray-900 hover:bg-gray-100"
        >
          Copy Summary
        </Button>
        <Button
          variant="outline"
          size="sm"
          onClick={handleDownloadJson}
          className="h-8 rounded-full bg-white px-3 border-gray-200 text-gray-900 hover:bg-gray-100"
        >
          Export Data
        </Button>
        {/*confirmation message when copy is successful*/}
        {copied && <span className="text-xs text-gray-500">Copied ✓</span>}
      </div>
    </div>
  );
}
