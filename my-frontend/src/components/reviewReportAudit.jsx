import React, { useState, useMemo, useEffect } from 'react';
import { ThreeDots } from 'react-loader-spinner';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism'; // VS Code dark theme
import Loader from './loader';
import { ChevronDownIcon, ChevronUpIcon } from 'lucide-react';

// --- 4. Main Audit Table Component ---
const ReviewReportAudit = () => {
  const [currentPage, setCurrentPage] = useState(1);
  const [modalOpen, setModalOpen] = useState(false);
  const [selectedLog, setSelectedLog] = useState(null);
  const [audit, setAudit] = useState([]);
  const [loading, setLoading] = useState(true);
  const [teachers, setTeachers] = useState([]);
  const [reports, setReports] = useState([]);
  const [roles, setRoles] = useState([]);

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
        const [workFlowResponse, teachersResponse, reportsResponse, rolesResponse] = await Promise.all
        ([
          fetch(`${backend_url}/teachers/workflowAudit`),
          fetch(`${backend_url}/teachers/`),
          fetch(`${backend_url}/teachers/getWorkflow_all`),
          fetch(`${backend_url}/roles`)
        ])

        if (!workFlowResponse.ok) throw new Error(`HTTP error! status: ${workFlowResponse.status}`);
        if (!teachersResponse.ok) throw new Error(`HTTP error! status: ${teachersResponse.status}`);
        if (!reportsResponse.ok) throw new Error(`HTTP error! status: ${reportsResponse.status}`);
        if (!rolesResponse.ok) throw new Error(`HTTP error! status: ${rolesResponse.status}`);

        const workFlowData = await workFlowResponse.json();
        const teachersData = await teachersResponse.json();
        const reportsData = await reportsResponse.json();
        const rolesData = await rolesResponse.json();

        setAudit(workFlowData?.data);
        setTeachers(teachersData?.data);
        setReports(reportsData?.data);
        setRoles(rolesData?.data);
      } catch (error) {
        console.error('Failed to fetch dropdowns', error);
      } finally {
        setLoading(false);
      }
    }
    fetchData()
    }, []);

    const auditLogs = audit;

  //Pagination
  const recordsPerPage = 10;
  const totalPages = Math.ceil(auditLogs.length / recordsPerPage);
  
  console.log(auditLogs)

  const paginatedLogs = useMemo(() => {
    const startIndex = (currentPage - 1) * recordsPerPage;
    return auditLogs.slice(startIndex, startIndex + recordsPerPage);
  }, [currentPage,auditLogs]);

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
    }
  };

  return (
    (loading) ? (<Loader />) : (
      <div className="p-4">
      <h1 h1 className="text-2xl font-bold mb-4">Work-Flow Audit</h1>
      <div className="overflow-x-auto shadow-md sm:rounded-lg">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-100">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Timestamp</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Record Id</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Teacher Name</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Teacher Role</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Report</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Comments</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Created By</th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {(loading) ? <ThreeDots height="50" width="50" radius="9" color="#4fa94d"  /> : paginatedLogs.map((log) => (
              <tr key={log.id} className="hover:bg-gray-50">
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{new Date(log.created_at).toLocaleDateString('en-Gb')}</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{log.reviewId}</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{teachers.find((teacher) => teacher.teacherId === log.teacherId)?.teacherName}</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{roles.find((role) => role.roleId === teachers.find((teacher) => teacher.teacherId === log.teacherId)?.role)?.roleName}</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{reports.find((report) => report.reportId === log.reportId)?.comments}</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{audit.find((audit) => audit.reviewId === log.reviewId)?.comments}</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{teachers.find((teacher) => teacher.teacherId === log.teacherId)?.teacherName}</td>
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

export default ReviewReportAudit;