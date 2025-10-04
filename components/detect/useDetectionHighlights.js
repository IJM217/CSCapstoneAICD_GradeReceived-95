//useDetectionHighlights: custom hook that processes AI detection results into highlight segments for text display. does NOT manage component state or handle file operations.
"use client";
import { useMemo } from "react";

//takes the AI detection results and splits the text into parts - some parts are marked as AI-generated (highlighted in colour) and others are human-written (shown normally)
export function useDetectionHighlights(inputText, detectionResult, lastAnalyzedText) {
  //highlights assume half-open ranges [start, end); overlapping ranges merge with max score
  return useMemo(() => {
    //return empty array if no results or text has changed since analysis
    if (!detectionResult || !detectionResult.highlights || !Array.isArray(detectionResult.highlights) || !detectionResult.highlights.length || inputText.trim() !== lastAnalyzedText.trim()) {
      return [];
    }
    
    const highlights = detectionResult.highlights;
    
    //process each highlight range and clamp to text bounds to prevent out-of-range errors
    const ranges = highlights
      .map(highlight => ({
        start: Math.max(0, Math.min(inputText.length, highlight.startIndex ?? 0)), //clamp start to text bounds
        end: Math.max(0, Math.min(inputText.length, highlight.endIndex ?? 0)), //clamp end to text bounds
        score: typeof highlight.score === "number" ? highlight.score : undefined, //preserve score if valid
      }))
      .filter(range => range.end > range.start) //remove invalid ranges
      .sort((a, b) => (a.start - b.start) || (a.end - b.end)); //sort by position

    //merge overlapping or touching ranges to prevent gaps
    const mergedRanges = [];
    for (const range of ranges) {
      if (mergedRanges.length && range.start <= mergedRanges[mergedRanges.length - 1].end) {
        //merge with previous range if overlapping or touching
        const lastRange = mergedRanges[mergedRanges.length - 1];
        lastRange.end = Math.max(lastRange.end, range.end);
        lastRange.score = Math.max(lastRange.score ?? 0, range.score ?? 0); //use max score
      } else {
        //add as new range
        mergedRanges.push({ ...range });
      }
    }

    //create text segments for rendering with highlights
    const segments = [];
    let currentIndex = 0;
    for (const range of mergedRanges) {
      //add human-written text before highlight
      if (currentIndex < range.start) {
        segments.push({ text: inputText.slice(currentIndex, range.start), ai: false });
      }
      //add AI-generated highlighted text
      segments.push({ text: inputText.slice(range.start, range.end), ai: true, score: range.score });
      currentIndex = range.end;
    }
    //add remaining human-written text after last highlight
    if (currentIndex < inputText.length) {
      segments.push({ text: inputText.slice(currentIndex), ai: false });
    }
    return segments;
  }, [inputText, detectionResult, lastAnalyzedText]);
}
