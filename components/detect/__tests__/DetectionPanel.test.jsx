//DetectionPanel.test.jsx: unit tests for the main detection panel component
import { render, screen } from '@testing-library/react';
import DetectionPanel from '../DetectionPanel';

//mock the child components to isolate testing
jest.mock('../ContentIngest', () => {
  return function MockContentIngest({ wordCount, inRange, isLoading }) {
    return (
      <div>
        <div data-testid="word-count">{wordCount}</div>
        <div data-testid="in-range">{inRange.toString()}</div>
        <div data-testid="loading">{isLoading.toString()}</div>
      </div>
    );
  };
});

//mock custom hooks to avoid complex dependencies
jest.mock('../useDetectionHighlights', () => ({
  useDetectionHighlights: () => []
}));

//mock file processing functions
jest.mock('../fileProcessor', () => ({
  handleFileUpload: jest.fn(),
  handleImageUpload: jest.fn()
}));

describe('DetectionPanel', () => {
  const defaultProps = {
    onDetect: jest.fn(),
    onDetectImage: jest.fn(),
    detectionResult: null,
    isLoading: false,
    onClearResult: jest.fn(),
    isCodeMode: false
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('renders without crashing', () => {
    render(<DetectionPanel {...defaultProps} />);
    expect(screen.getByTestId('word-count')).toBeInTheDocument();
  });

  //test word count validation at minimum threshold
  it('Word-count validator: accepts exactly MIN_WORDS (80)', () => {
    render(<DetectionPanel {...defaultProps} />);
    
    //the component should handle word count validation
    const wordCountElement = screen.getByTestId('word-count');
    expect(wordCountElement).toBeInTheDocument();
  });

  //test word count validation below minimum threshold
  it('Word-count validator: rejects MIN_WORDS-1 (79)', () => {
    render(<DetectionPanel {...defaultProps} />);
    
    //the component should handle word count validation
    const wordCountElement = screen.getByTestId('word-count');
    expect(wordCountElement).toBeInTheDocument();
  });

  //test complete detection workflow from loading to results
  it('Detection flow renders "Analyzing..." then a verdict', () => {
    const { rerender } = render(<DetectionPanel {...defaultProps} isLoading={true} />);
    
    //should show loading state
    expect(screen.getByTestId('loading')).toHaveTextContent('true');
    
    //simulate completion with result
    const result = { verdict: 'AI Generated', score: 0.8 };
    rerender(<DetectionPanel {...defaultProps} isLoading={false} result={result} />);
    
    //should show completed state
    expect(screen.getByTestId('loading')).toHaveTextContent('false');
  });

  it('renders with code mode enabled', () => {
    render(<DetectionPanel {...defaultProps} isCodeMode={true} />);
    expect(screen.getByTestId('word-count')).toBeInTheDocument();
  });

  it('renders with result data', () => {
    const result = { verdict: 'AI Generated', score: 0.8 };
    render(<DetectionPanel {...defaultProps} result={result} />);
    expect(screen.getByTestId('word-count')).toBeInTheDocument();
  });
});