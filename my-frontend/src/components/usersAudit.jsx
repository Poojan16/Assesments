import React, { useState, useMemo, useEffect } from 'react';
import { ThreeDots } from 'react-loader-spinner';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism'; // VS Code dark theme
import Loader from './loader';
import { ChevronDownIcon, ChevronUpIcon } from 'lucide-react';
import {useSelector} from 'react-redux';
import { useNavigate } from 'react-router-dom';

// --- 4. Main Audit Table Component ---
const UserAudit = () => {
  const [currentPage, setCurrentPage] = useState(1);
  const [modalOpen, setModalOpen] = useState(false);
  const [selectedLog, setSelectedLog] = useState(null);
  const [audit, setAudit] = useState([]);
  const [loading, setLoading] = useState(true);
  const [recordsPerPage, setRecordsPerPage] = useState(10);
  const [offset, setOffset] = useState(0);
  const navigate = useNavigate();
  const { token } = useSelector((state) => state.auth);
  const [paginatedData, setPaginatedData] = useState([]);
  const backend_url = process.env.REACT_APP_BACKEND_URL;

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

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      try {
        const response = await fetch(`${backend_url}/audit/user_audits?limit=${recordsPerPage}&offset=${offset}`);
        const data = await response.json();
        if (!response.ok) { // Check for HTTP errors
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const auditData = data?.data
        for (const audit of auditData) {
            const userLongins = await fetch(`${backend_url}/sessions?sessionId=${audit.sessionId}`)
            const userLoginData = await userLongins.json();
            const LoginData = userLoginData?.data
            const deviceDetails = LoginData?.data?.deviceInfo;
            const userData = await fetch(`${backend_url}/users/id?user_id=${audit.user_id}`);
            const user = await userData.json();
            audit.name = (user?.data)?.userName
            audit.email = (user?.data)?.userEmail
            audit.ip = deviceDetails?.ip
            audit.device =  deviceDetails?.device
            audit.os = deviceDetails?.os
            audit.browser = deviceDetails?.browser
            audit.login_time = new Date(LoginData?.data?.loginTime).toLocaleTimeString()
            audit.status = LoginData?.data?.isActive ? 'Active' : 'Inactive'
            audit.user_id = (user?.data)?.userName
        }
        console.log(auditData)
        setAudit(auditData);
        setPaginatedData(data?.pagination);
      } catch (error) {
        console.error('Failed to fetch dropdowns', error);
      } finally {
        setLoading(false);
      }
    }
    fetchData()
    }, [recordsPerPage, offset]);


    const auditLogs = audit;
    console.log(auditLogs)

  //Pagination
  const totalPages = paginatedData?.total_pages;

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
              ← Back 
            </button>
        </div>
      <h1 h1 className="text-2xl font-bold mb-4">User Audit</h1>
      <div className="overflow-x-auto shadow-md sm:rounded-lg">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-100">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Timestamp</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">User Name</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Email</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">IP</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Device</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">OS</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Browser</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Login Time</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Activity</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Status</th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {(loading) ? <ThreeDots height="50" width="50" radius="9" color="#4fa94d"  /> : auditLogs.map((log) => (
              <tr key={log.id} className="hover:bg-gray-50">
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{new Date(log.created_at).toLocaleDateString('en-Gb')}</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{log.name}</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{log.email}</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{log.ip}</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{log.device}</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{log.os}</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{log.browser}</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{log.login_time}</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{log.activity}</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{log.status}</td>
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
                {(  (currentPage - 1) * recordsPerPage ) + auditLogs.length}
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

export default UserAudit;