import React, { useState, useMemo, useEffect } from 'react';
import { ThreeDots } from 'react-loader-spinner';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism'; // VS Code dark theme
import Loader from './loader';
import { useNavigate } from 'react-router-dom';
import { ChevronDownIcon, ChevronUpIcon } from 'lucide-react';

// --- 4. Main Audit Table Component ---
const AuditTable = () => {
  const [currentPage, setCurrentPage] = useState(1);
  const [modalOpen, setModalOpen] = useState(false);
  const [selectedLog, setSelectedLog] = useState(null);
  const [audit, setAudit] = useState([]);
  const [loading, setLoading] = useState(true);
  const [offset, setOffset] = useState(0);
  const [recordsPerPage, setRecordsPerPage] = useState(10);
  const navigate = useNavigate();
  const [paginatedData, setPaginatedData] = useState([]);
  const [total_records, setTotalRecords] = useState(0);

  useEffect(() => {
    // Simulate a loading process (e.g., data fetching, component mounting)
    const timer = setTimeout(() => {
      setLoading(false); // Hide loader after a delay
    }, 1000); // Adjust delay as needed

    return () => clearTimeout(timer); // Cleanup timer on unmount
  }, []);

  useEffect(() => {
    document.title = 'Audit Trail'; // Set the desired title here
  }, []);

  const backend_url = process.env.REACT_APP_BACKEND_URL;

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      try {
        const response = await fetch(`${backend_url}/audit/?limit=${recordsPerPage}&offset=${offset}`);
        const data = await response.json();
        if (!response.ok) { // Check for HTTP errors
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        setAudit(data?.data);
        setPaginatedData(data?.pagination);
        setTotalRecords(paginatedData?.total_records);
        console.log(data.changedBy)
        for (const audit of data) {
          const teacher = await fetch(`${backend_url}/teachers/id?teacherId=${Number(audit.changedBy)}`);
          const teacherData = await teacher.json();
          audit.changedBy = teacherData.teacherName
        }
      } catch (error) {
        console.error('Failed to fetch dropdowns', error);
      } finally {
        setLoading(false);
      }
    }
    fetchData()
    }, [recordsPerPage, offset,backend_url]);

    const auditLogs = audit;

  //Pagination
  const totalPages = paginatedData?.total_pages;
  
  console.log(auditLogs)

  // const paginatedLogs = useMemo(() => {
  //   const startIndex = (currentPage - 1) * recordsPerPage;
  //   return auditLogs.slice(startIndex, startIndex + recordsPerPage);
  // }, [currentPage,auditLogs]);

  const openModal = (log) => {
    setSelectedLog(log);
    setModalOpen(true);
  };

  const closeModal = () => {
    setSelectedLog(null);
    setModalOpen(false);
  };

  const goToPage = (page) => {
    if (page >= 1 && page <= totalPages) {
      setCurrentPage(page);
      setOffset((page - 1) * recordsPerPage);
    }
  };

  return (
    (loading) ? (<Loader />) : (
      <div className="p-4">
        <div>
        <button
              onClick={() => navigate(-1)}
              className="text-blue-600 hover:text-blue-700 font-medium mb-4 flex items-center gap-2"
            >
              ← Back to Dashboard
            </button>
        </div>
      <div className='flex justify-between'>
        <h1 h1 className="text-2xl font-bold mb-4">Audit Trail</h1>
        <div className='flex gap-2'>
          <button onClick={() => navigate('/reviewAudit')} className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded mb-5"><b>Work-Flow Audit</b></button>
          <button onClick={() => navigate('/user-audit')} className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded mb-5"><b>User Audit</b></button>
        </div>
      </div>
      <div className="overflow-x-auto shadow-md sm:rounded-lg">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-100">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Timestamp</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Table Name</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Record Id</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Field Name</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Old Value</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">New Value</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Reason</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Modified By</th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {(loading) ? <ThreeDots height="50" width="50" radius="9" color="#4fa94d"  /> : audit.map((log) => (
              <tr key={log.id} className="hover:bg-gray-50">
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{new Date(log.created_at).toLocaleDateString('en-Gb')}</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{log.tableName}</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{log.fieldId}</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{log.fieldName}</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{(log.oldValue) === '1' ? 'Active' : 'Inactive'}</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{(log.newValue) === '1' ? 'Active' : 'Inactive'}</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{log.reason}</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{log.changedBy}</td>
                <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Pagination Controls */}
      <div className="bg-white px-4 py-3 flex items-center justify-between border-t border-gray-200 sm:px-6">
        <div className="flex-1 flex justify-between sm:hidden">
          <button
            onClick={() => goToPage(currentPage - 1)}
            disabled={currentPage === 1}
            className="relative inline-flex items-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 disabled:opacity-50"
          >
            Previous
          </button>
          <button
            onClick={() => goToPage(currentPage + 1)}
            disabled={currentPage === totalPages}
            className="ml-3 relative inline-flex items-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 disabled:opacity-50"
          >
            Next
          </button>
        </div>
        <div className="hidden sm:flex-1 sm:flex sm:items-center sm:justify-between">
          <div>
            <p className="text-sm text-gray-700">
              Showing <span className="font-medium">{(currentPage - 1) * recordsPerPage + 1}</span> to{' '}
              <span className="font-medium">
                {Math.min(currentPage * recordsPerPage, auditLogs.length)}
              </span>{' '}
              of <span className="font-medium">{auditLogs.length}</span> results
            </p>
          </div>
          <div>
            <nav className="relative z-0 inline-flex rounded-md shadow-sm -space-x-px" aria-label="Pagination">
              <button
                onClick={() => goToPage(currentPage - 1)}
                disabled={currentPage === 1}
                className="relative inline-flex items-center px-2 py-2 rounded-l-md border border-gray-300 bg-white text-sm font-medium text-gray-500 hover:bg-gray-50 disabled:opacity-50"
              >
                <span className="sr-only">Previous</span>
                <ChevronDownIcon className="h-5 w-5 rotate-90" />
              </button>
              {/* Simple page number display */}
              <span className="relative inline-flex items-center px-4 py-2 border border-gray-300 bg-blue-500 text-sm font-medium text-white">
                {currentPage}
              </span>
              <button
                onClick={() => goToPage(currentPage + 1)}
                disabled={currentPage === totalPages}
                className="relative inline-flex items-center px-2 py-2 rounded-r-md border border-gray-300 bg-white text-sm font-medium text-gray-500 hover:bg-gray-50 disabled:opacity-50"
              >
                <span className="sr-only">Next</span>
                <ChevronUpIcon className="h-5 w-5 rotate-90" />
              </button>
            </nav>
          </div>
        </div>
      </div>
    </div>
    )
  );
};

export default AuditTable;