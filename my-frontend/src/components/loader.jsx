// Loader.js
import React from 'react';
import { Circles, ThreeDots } from 'react-loader-spinner';


const Loader = () => {
    return (
      <div className="flex items-center justify-center min-h-screen bg-gray-100">
        <div className="flex flex-col items-center space-y-4">
          {/* Spinner */}
          {/* <div class="w-16 h-16 border-4 border-t-4 border-blue-500 border-solid rounded-full animate-spin"></div> */}
          <Circles color='#2563EB'/>
          {/* Loading Text (Optional) */}
          <p className="text-lg font-semibold text-gray-700">Loading...</p>
        </div>
      </div>
    );
  };
  
  export default Loader;