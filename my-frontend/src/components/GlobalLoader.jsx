// src/components/GlobalLoader.js
import React from 'react';
import { useLoading } from '../loadingContext';
import { ThreeDots } from 'react-loader-spinner';

const GlobalLoader = () => {
  const { isLoading } = useLoading();

  if (!isLoading) return null;

  return (
    <div className="loader-overlay">
      <div className="loader-spinner">
        <ThreeDots color="#2563EB" height={80} width={80} />  
      </div> {/* Use a spinner library or CSS */}
    </div>
  );
};

export default GlobalLoader;
