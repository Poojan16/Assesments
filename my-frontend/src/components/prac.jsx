// HeadTeacherReport.jsx

import React from 'react';

const ReportContainer = () => {
  // Sample data for the report
  const reportData = [
    { subject: 'Mathematics', averageScore: 78, passRate: 85, topStudent: 'Alice Smith' },
    { subject: 'English Literature', averageScore: 82, passRate: 90, topStudent: 'Bob Johnson' },
    { subject: 'Science', averageScore: 75, passRate: 80, topStudent: 'Charlie Brown' },
    { subject: 'History', averageScore: 88, passRate: 95, topStudent: 'Diana Prince' },
  ];
  const className = "Year 7A";
  const classSize = 30;
  const headTeacherName = "Mr. Albus Dumbledore";

  return (
    <div className="min-h-screen bg-gray-100 p-4 sm:p-8">
      <div className="max-w-6xl mx-auto bg-white shadow-xl rounded-lg p-6">
        
        {/* Header Section */}
        <header className="border-b pb-4 mb-6">
          <h1 className="text-3xl font-bold text-gray-800">Head Teacher's Summary Report</h1>
          <p className="text-gray-600">Academic Year: 2024-2025</p>
        </header>

        {/* Summary Information and Filters */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <div className="bg-blue-50 p-4 rounded-lg shadow-sm">
            <p className="text-sm font-medium text-blue-700">Class: <span className="font-semibold">{className}</span></p>
            <p className="text-sm font-medium text-blue-700">Class Size: <span className="font-semibold">{classSize} students</span></p>
          </div>
          <div className="bg-green-50 p-4 rounded-lg shadow-sm">
            <p className="text-sm font-medium text-green-700">Head Teacher: <span className="font-semibold">{headTeacherName}</span></p>
          </div>
          <div className="bg-yellow-50 p-4 rounded-lg shadow-sm">
            <label htmlFor="subjectFilter" className="block text-sm font-medium text-yellow-700 mb-1">Filter by Subject:</label>
            <select id="subjectFilter" className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-yellow-500 focus:border-yellow-500 sm:text-sm rounded-md">
              <option>All Subjects</option>
              {reportData.map((item, index) => (
                <option key={index}>{item.subject}</option>
              ))}
            </select>
          </div>
        </div>

        {/* Data Table Section */}
        <div className="overflow-x-auto shadow ring-1 ring-black ring-opacity-5 md:rounded-lg">
          <table className="min-w-full divide-y divide-gray-300">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Subject</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Average Score (%)</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Pass Rate (%)</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Top Student</th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {reportData.map((item, index) => (
                <tr key={index} className={index % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">{item.subject}</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{item.averageScore}</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{item.passRate}</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{item.topStudent}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Footer/Disclaimer Section */}
        <footer className="mt-8 pt-4 border-t text-xs text-gray-500">
          <p>This report provides a summary of academic performance for the specified class. Data is accurate as of the generation date.</p>
        </footer>
      </div>
    </div>
  );
};

export default ReportContainer;