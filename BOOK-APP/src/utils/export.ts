/**
 * Export Utility
 * Handles exporting annotations, highlights, and notes in various formats
 * @module utils/export
 */

import * as FileSystem from 'expo-file-system';
import * as Sharing from 'expo-sharing';
import { logger } from './logger';
import { formatDate } from './format';

// ==================== TYPES ====================

export interface ExportOptions {
  format: 'txt' | 'json' | 'md' | 'html' | 'csv';
  includeMetadata?: boolean;
  groupByPage?: boolean;
}

export interface AnnotationExportData {
  id: string;
  type: 'highlight' | 'note' | 'drawing';
  content: string;
  pageNumber: number;
  color: string;
  createdAt: string;
  updatedAt: string;
}

export interface BookExportData {
  bookId: string;
  title: string;
  author?: string;
  annotations: AnnotationExportData[];
  exportedAt: string;
}

// ==================== EXPORT FUNCTIONS ====================

/**
 * Exports annotations to a file
 * @param data - Book and annotation data
 * @param options - Export options
 * @returns Path to exported file
 */
export async function exportAnnotations(
  data: BookExportData,
  options: ExportOptions
): Promise<string> {
  try {
    let content: string;
    let filename: string;
    let mimeType: string;

    switch (options.format) {
      case 'txt':
        content = exportToText(data, options);
        filename = `${sanitizeFilename(data.title)}_annotations.txt`;
        mimeType = 'text/plain';
        break;

      case 'json':
        content = exportToJSON(data, options);
        filename = `${sanitizeFilename(data.title)}_annotations.json`;
        mimeType = 'application/json';
        break;

      case 'md':
        content = exportToMarkdown(data, options);
        filename = `${sanitizeFilename(data.title)}_annotations.md`;
        mimeType = 'text/markdown';
        break;

      case 'html':
        content = exportToHTML(data, options);
        filename = `${sanitizeFilename(data.title)}_annotations.html`;
        mimeType = 'text/html';
        break;

      case 'csv':
        content = exportToCSV(data, options);
        filename = `${sanitizeFilename(data.title)}_annotations.csv`;
        mimeType = 'text/csv';
        break;

      default:
        throw new Error(`Unsupported format: ${options.format}`);
    }

    // Write to file
    const fileUri = `${FileSystem.documentDirectory}${filename}`;
    await FileSystem.writeAsStringAsync(fileUri, content, {
      encoding: FileSystem.EncodingType.UTF8,
    });

    logger.info('Annotations exported', { filename, format: options.format });

    return fileUri;
  } catch (error) {
    logger.error('Failed to export annotations', error);
    throw error;
  }
}

/**
 * Exports and shares annotations
 * @param data - Book and annotation data
 * @param options - Export options
 */
export async function exportAndShare(
  data: BookExportData,
  options: ExportOptions
): Promise<void> {
  try {
    const fileUri = await exportAnnotations(data, options);

    // Share the file
    const canShare = await Sharing.isAvailableAsync();
    if (canShare) {
      await Sharing.shareAsync(fileUri, {
        dialogTitle: 'Share Annotations',
      });
    } else {
      throw new Error('Sharing is not available on this platform');
    }

    logger.info('Annotations shared', { format: options.format });
  } catch (error) {
    logger.error('Failed to share annotations', error);
    throw error;
  }
}

// ==================== FORMAT CONVERTERS ====================

/**
 * Exports to plain text format
 */
function exportToText(data: BookExportData, options: ExportOptions): string {
  let content = '';

  // Header
  if (options.includeMetadata) {
    content += `Book: ${data.title}\n`;
    if (data.author) content += `Author: ${data.author}\n`;
    content += `Exported: ${formatDate(data.exportedAt)}\n`;
    content += `Total Annotations: ${data.annotations.length}\n`;
    content += '\n' + '='.repeat(60) + '\n\n';
  }

  // Annotations
  if (options.groupByPage) {
    const grouped = groupAnnotationsByPage(data.annotations);
    Object.keys(grouped)
      .sort((a, b) => Number(a) - Number(b))
      .forEach((page) => {
        content += `Page ${page}\n`;
        content += '-'.repeat(40) + '\n';
        grouped[page].forEach((annotation) => {
          content += formatAnnotationText(annotation);
          content += '\n\n';
        });
      });
  } else {
    data.annotations.forEach((annotation) => {
      content += formatAnnotationText(annotation);
      content += '\n\n';
    });
  }

  return content;
}

/**
 * Exports to JSON format
 */
function exportToJSON(data: BookExportData, options: ExportOptions): string {
  const exportData = {
    book: {
      id: data.bookId,
      title: data.title,
      author: data.author,
    },
    exportedAt: data.exportedAt,
    totalAnnotations: data.annotations.length,
    annotations: data.annotations.map((a) => ({
      id: a.id,
      type: a.type,
      content: a.content,
      pageNumber: a.pageNumber,
      color: a.color,
      createdAt: a.createdAt,
      updatedAt: a.updatedAt,
    })),
  };

  return JSON.stringify(exportData, null, 2);
}

/**
 * Exports to Markdown format
 */
function exportToMarkdown(data: BookExportData, options: ExportOptions): string {
  let content = '';

  // Header
  if (options.includeMetadata) {
    content += `# ${data.title}\n\n`;
    if (data.author) content += `**Author:** ${data.author}\n\n`;
    content += `**Exported:** ${formatDate(data.exportedAt)}\n`;
    content += `**Total Annotations:** ${data.annotations.length}\n\n`;
    content += '---\n\n';
  }

  // Annotations
  if (options.groupByPage) {
    const grouped = groupAnnotationsByPage(data.annotations);
    Object.keys(grouped)
      .sort((a, b) => Number(a) - Number(b))
      .forEach((page) => {
        content += `## Page ${page}\n\n`;
        grouped[page].forEach((annotation) => {
          content += formatAnnotationMarkdown(annotation);
          content += '\n\n';
        });
      });
  } else {
    data.annotations.forEach((annotation) => {
      content += formatAnnotationMarkdown(annotation);
      content += '\n\n';
    });
  }

  return content;
}

/**
 * Exports to HTML format
 */
function exportToHTML(data: BookExportData, options: ExportOptions): string {
  let html = `
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>${data.title} - Annotations</title>
  <style>
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      max-width: 800px;
      margin: 0 auto;
      padding: 20px;
      line-height: 1.6;
      color: #333;
    }
    h1 { color: #1F2937; }
    h2 { color: #4B5563; margin-top: 2rem; }
    .metadata { color: #6B7280; margin-bottom: 2rem; }
    .annotation {
      margin: 1.5rem 0;
      padding: 1rem;
      border-left: 4px solid #6366F1;
      background: #F9FAFB;
      border-radius: 4px;
    }
    .annotation-header {
      font-weight: 600;
      color: #4B5563;
      margin-bottom: 0.5rem;
    }
    .annotation-content {
      white-space: pre-wrap;
      margin: 0.5rem 0;
    }
    .annotation-meta {
      font-size: 0.875rem;
      color: #9CA3AF;
    }
    .highlight { background: #FEF08A; padding: 2px 4px; }
    .note { background: #DBEAFE; padding: 2px 4px; }
  </style>
</head>
<body>
`;

  // Header
  if (options.includeMetadata) {
    html += `  <h1>${data.title}</h1>\n`;
    if (data.author) html += `  <p class="metadata"><strong>Author:</strong> ${data.author}</p>\n`;
    html += `  <p class="metadata"><strong>Exported:</strong> ${formatDate(data.exportedAt)}</p>\n`;
    html += `  <p class="metadata"><strong>Total Annotations:</strong> ${data.annotations.length}</p>\n`;
    html += `  <hr>\n`;
  }

  // Annotations
  if (options.groupByPage) {
    const grouped = groupAnnotationsByPage(data.annotations);
    Object.keys(grouped)
      .sort((a, b) => Number(a) - Number(b))
      .forEach((page) => {
        html += `  <h2>Page ${page}</h2>\n`;
        grouped[page].forEach((annotation) => {
          html += formatAnnotationHTML(annotation);
        });
      });
  } else {
    data.annotations.forEach((annotation) => {
      html += formatAnnotationHTML(annotation);
    });
  }

  html += `
</body>
</html>`;

  return html;
}

/**
 * Exports to CSV format
 */
function exportToCSV(data: BookExportData, options: ExportOptions): string {
  let csv = 'Type,Page,Content,Color,Created,Updated\n';

  data.annotations.forEach((annotation) => {
    const row = [
      annotation.type,
      annotation.pageNumber,
      `"${annotation.content.replace(/"/g, '""')}"`, // Escape quotes
      annotation.color,
      formatDate(annotation.createdAt),
      formatDate(annotation.updatedAt),
    ].join(',');

    csv += row + '\n';
  });

  return csv;
}

// ==================== HELPER FUNCTIONS ====================

/**
 * Groups annotations by page number
 */
function groupAnnotationsByPage(
  annotations: AnnotationExportData[]
): Record<number, AnnotationExportData[]> {
  return annotations.reduce((acc, annotation) => {
    if (!acc[annotation.pageNumber]) {
      acc[annotation.pageNumber] = [];
    }
    acc[annotation.pageNumber].push(annotation);
    return acc;
  }, {} as Record<number, AnnotationExportData[]>);
}

/**
 * Formats annotation for text output
 */
function formatAnnotationText(annotation: AnnotationExportData): string {
  return `[${annotation.type.toUpperCase()}] Page ${annotation.pageNumber}\n${annotation.content}\nCreated: ${formatDate(annotation.createdAt)}`;
}

/**
 * Formats annotation for Markdown output
 */
function formatAnnotationMarkdown(annotation: AnnotationExportData): string {
  return `### ${annotation.type.charAt(0).toUpperCase() + annotation.type.slice(1)} (Page ${annotation.pageNumber})\n\n${annotation.content}\n\n*Created: ${formatDate(annotation.createdAt)}*`;
}

/**
 * Formats annotation for HTML output
 */
function formatAnnotationHTML(annotation: AnnotationExportData): string {
  return `  <div class="annotation">
    <div class="annotation-header">${annotation.type.charAt(0).toUpperCase() + annotation.type.slice(1)} - Page ${annotation.pageNumber}</div>
    <div class="annotation-content ${annotation.type}">${annotation.content}</div>
    <div class="annotation-meta">Created: ${formatDate(annotation.createdAt)}</div>
  </div>\n`;
}

/**
 * Sanitizes filename for safe file system usage
 */
function sanitizeFilename(filename: string): string {
  return filename
    .replace(/[^a-zA-Z0-9_-]/g, '_')
    .replace(/_+/g, '_')
    .toLowerCase();
}

// ==================== EXPORT ====================

export default {
  exportAnnotations,
  exportAndShare,
};
