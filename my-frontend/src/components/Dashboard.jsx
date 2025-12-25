import React, { useState } from 'react';
import { useForm } from 'react-hook-form';
import * as XLSX from 'xlsx'; // Import the xlsx library




const AdminDashboard = () => {
  const { register, handleSubmit } = useForm();
  const [successMessage, setSuccessMessage] = useState('');
  const [excelData, setExcelData] = useState(null);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        const data = new Uint8Array(e.target.result);
        const workbook = XLSX.read(data, { type: 'array' });
        const sheetName = workbook.SheetNames[0]; // Get the first sheet
        const worksheet = workbook.Sheets[sheetName];
        const jsonData = XLSX.utils.sheet_to_json(worksheet); // Convert to JSON
        setExcelData(jsonData);
      };
      reader.readAsArrayBuffer(file);
    }
  };

  // const purify = DOMpurify.sanitize(excelData)
  // console.log(purify)

  const onSubmit = (data) => {
    // Handle form submission here
    console.log(data);

    const url = 'http://127.0.0.1:8000/importExcel';

    const method = 'POST';

    fetch(url, {
      method,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(excelData),
    })
      .then((response) => response.json())
      .then((data) => {
        console.log(data);
      })
      .catch((error) => {
        console.error('Error:', error);
      });
    setSuccessMessage('Task updated successfully!');
  };
  
  
  const handleImportExcel = () => {
    // Handle import logic here
    alert('Importing Excel...');
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-100">
      <div className="grid grid-col-3 bg-white p-8 rounded-lg shadow-md w-full max-w-lg">
        <h2 className="text-2xl font-bold mb-6 text-center">Update Task</h2>

        <p className='m-4 text-green-600'>{successMessage}</p>

        <form onSubmit={handleSubmit(onSubmit)} encType="multipart/form-data">
          {/* Title */}
          <div className="mb-4">
            <label className="block text-gray-700 mb-2">Attach Excel</label>
            <input
              {...register('file')}
              type="file"
              accept='.xlsx, .xls'
              onChange={handleFileChange}
              className="w-full px-4 py-2 border rounded-lg bg-gray-100 "
            />
          </div>

          {/* Submit Button */}
          <button
            type="submit"
            className="w-full bg-blue-500 hover:bg-blue-600 text-white py-2 rounded-lg"
          >
            Import Excel
          </button>
          {excelData && (
                <pre>{JSON.stringify(excelData, null, 2)}</pre> // Display data for verification
              )}
        </form>
      </div>
    </div>
  );
};

export default AdminDashboard;