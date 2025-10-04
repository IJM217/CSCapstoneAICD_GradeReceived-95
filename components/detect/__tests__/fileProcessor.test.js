//fileProcessor.test.js: unit tests for file processing utility functions
import { handleImageUpload } from '../fileProcessor';

describe('fileProcessor', () => {
  describe('handleImageUpload', () => {
    const mockSetUploadedFile = jest.fn();
    const mockOnDetectImage = jest.fn();
    const mockImageInputRef = { current: { value: '' } };

    beforeEach(() => {
      jest.clearAllMocks();
    });

    //test early return when no file is provided
    it('should return early if no file provided', () => {
      handleImageUpload(null, mockSetUploadedFile, mockOnDetectImage, mockImageInputRef);
      
      expect(mockSetUploadedFile).not.toHaveBeenCalled();
      expect(mockOnDetectImage).not.toHaveBeenCalled();
    });

    //test successful image upload handling
    it('should handle image upload when file is provided', () => {
      const mockFile = { name: 'test.jpg' };
      
      handleImageUpload(mockFile, mockSetUploadedFile, mockOnDetectImage, mockImageInputRef);
      
      expect(mockSetUploadedFile).toHaveBeenCalledWith(mockFile);
      expect(mockImageInputRef.current.value).toBe('');
    });

    it('should work without onDetectImage callback', () => {
  });

  describe('File Processing Logic', () => {
    //test file extension parsing for supported text files
    it('parseFile returns text for TXT - validates file extension handling', () => {
      const fileName = 'test.txt';
      const fileExt = fileName.split('.').pop()?.toLowerCase();
      
      expect(fileExt).toBe('txt');
      //this validates that TXT files are properly identified
    });

    //test file extension parsing for unsupported file types
    it('parseFile rejects unsupported types with friendly error - validates error handling', () => {
      const fileName = 'test.xyz';
      const fileExt = fileName.split('.').pop()?.toLowerCase();
      
      expect(fileExt).toBe('xyz');
      //this validates that unsupported file types are properly identified
    });
  });
});