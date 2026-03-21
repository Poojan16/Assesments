import React, { useState, useEffect } from 'react';
import { Document, Page } from 'react-pdf';
import { pdfjs } from 'react-pdf';

pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  'pdfjs-dist/build/pdf.worker.min.mjs',
  import.meta.url,
).toString();

function MyPdfViewer({ pdfBlobUrl}) {
  const [numPages, setNumPages] = useState(null);

  const onDocumentLoadSuccess = ({ numPages }) => {
    setNumPages(numPages);
  };

  return (
    <div>
      {pdfBlobUrl && (
        <Document
          file={pdfBlobUrl}
          onLoadSuccess={onDocumentLoadSuccess}
        >
          {Array.from(new Array(numPages), (el, index) => (
            <Page key={`page_${index + 1}`} pageNumber={index + 1} className={'max-w-lg p-4 mx-auto mt-4 border border-gray-300 rounded-lg'} renderTextLayer={false}
            renderAnnotationLayer={false}
            />
          ))}
        </Document>
      )}
      {!pdfBlobUrl && <p>Loading PDF...</p>}
    </div>
  );
}

export default MyPdfViewer;