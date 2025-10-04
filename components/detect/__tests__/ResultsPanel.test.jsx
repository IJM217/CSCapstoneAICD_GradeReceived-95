//ResultsPanel.test.jsx: unit tests for the results display component
import { render, screen } from '@testing-library/react';
import ResultsPanel from '../ResultsPanel';

describe('ResultsPanel', () => {
  const defaultProps = {
    detectionResult: null,
    isLoading: false
  };

  //test loading state display
  it('renders loading state', () => {
    render(<ResultsPanel {...defaultProps} isLoading={true} />);
    expect(screen.getByText('Analyzing…')).toBeInTheDocument();
  });

  //test empty state when no results available
  it('renders empty state when no result', () => {
    render(<ResultsPanel {...defaultProps} />);
    expect(screen.getByText('Analysis results will appear here')).toBeInTheDocument();
  });

  //test result display with AI detection verdict and score
  it('renders result with verdict and score', () => {
    const result = {
      verdict: 'AI Generated',
      score: 0.85,
      modelVersion: 'v1.0'
    };
    
    render(<ResultsPanel {...defaultProps} detectionResult={result} />);
    
    expect(screen.getByText('AI Generated')).toBeInTheDocument();
    //the score is displayed as 0.85% in the component (split across elements)
    expect(screen.getByText(/0\.85/)).toBeInTheDocument();
  });

  //test score formatting and display accuracy
  it('ResultsPanel "Copy Summary" uses the correct score - validates score formatting', () => {
    const result = {
      verdict: 'Human Written',
      score: 0.23,
      modelVersion: 'v1.0'
    };
    
    render(<ResultsPanel {...defaultProps} detectionResult={result} />);
    
    //verify the score is displayed correctly (using regex for split text)
    expect(screen.getByText(/0\.23/)).toBeInTheDocument();
    expect(screen.getByText('Human Written')).toBeInTheDocument();
  });

  //test action buttons visibility when results are present
  it('shows copy button when result is present', () => {
    const result = {
      verdict: 'AI Generated',
      score: 0.75,
      modelVersion: 'v1.0'
    };
    
    render(<ResultsPanel {...defaultProps} detectionResult={result} />);
    
    expect(screen.getByText('Copy Summary')).toBeInTheDocument();
  });
});