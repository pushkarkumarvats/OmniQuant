import { formatBytes, formatDate, formatPageRange, truncateText } from '@/utils/format';

describe('format utils', () => {
  describe('formatBytes', () => {
    it('should format bytes correctly', () => {
      expect(formatBytes(0)).toBe('0 Bytes');
      expect(formatBytes(1024)).toBe('1 KB');
      expect(formatBytes(1048576)).toBe('1 MB');
      expect(formatBytes(1073741824)).toBe('1 GB');
    });

    it('should handle decimal places', () => {
      expect(formatBytes(1536, 1)).toBe('1.5 KB');
      expect(formatBytes(1536, 0)).toBe('2 KB');
    });
  });

  describe('formatPageRange', () => {
    it('should format single page', () => {
      expect(formatPageRange(5)).toBe('Page 5');
    });

    it('should format page range', () => {
      expect(formatPageRange(5, 10)).toBe('Pages 5-10');
    });

    it('should handle same start and end', () => {
      expect(formatPageRange(5, 5)).toBe('Page 5');
    });
  });

  describe('truncateText', () => {
    it('should not truncate short text', () => {
      expect(truncateText('Hello', 10)).toBe('Hello');
    });

    it('should truncate long text', () => {
      expect(truncateText('Hello World!', 8)).toBe('Hello...');
    });
  });
});
