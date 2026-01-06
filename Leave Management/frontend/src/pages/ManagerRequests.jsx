import { useState, useEffect } from 'react'
import { managerAPI } from '../services/api'
import DashboardLayout from '../layouts/DashboardLayout'

const ManagerRequests = () => {
  const [leaves, setLeaves] = useState([])
  const [loading, setLoading] = useState(true)
  const [selectedLeave, setSelectedLeave] = useState(null)
  const [showActionModal, setShowActionModal] = useState(false)
  const [actionData, setActionData] = useState({
    status: '',
    remarks: ''
  })
  const [actionLoading, setActionLoading] = useState(false)
  const [filters, setFilters] = useState({
    status: '',
    leave_type: '',
    start_date: '',
    end_date: '',
    employee_id: '',
    department: '',
  })
  const [pagination, setPagination] = useState({
    skip: 0,
    limit: 10,
    total: 0,
  })
  const [stats, setStats] = useState({
    total: 0,
    pending: 0,
    approved: 0,
    rejected: 0,
  })

  useEffect(() => {
    fetchLeaves()
    fetchStats()
  }, [filters, pagination.skip])

  const fetchLeaves = async () => {
    try {
      setLoading(true)
      const params = {
        ...filters,
        skip: pagination.skip,
        limit: pagination.limit,
      }
      Object.keys(params).forEach(key => {
        if (params[key] === '') {
          delete params[key]
        }
      })

      const response = await managerAPI.getAllLeaves(params)
      setLeaves(response.data.data || [])
      setPagination(prev => ({
        ...prev,
        total: response.data.total || 0,
      }))
    } catch (error) {
      console.error('Error fetching leaves:', error)
    } finally {
      setLoading(false)
    }
  }

  const fetchStats = async () => {
    try {
      const response = await managerAPI.getStats()
      if (response.data.success) {
        setStats(response.data.data)
      }
    } catch (error) {
      console.error('Error fetching stats:', error)
    }
  }

  const handleFilterChange = (name, value) => {
    setFilters(prev => ({
      ...prev,
      [name]: value,
    }))
    setPagination(prev => ({
      ...prev,
      skip: 0,
    }))
  }

  const clearFilters = () => {
    setFilters({
      status: '',
      leave_type: '',
      start_date: '',
      end_date: '',
      employee_id: '',
      department: '',
    })
    setPagination(prev => ({
      ...prev,
      skip: 0,
    }))
  }

  const openActionModal = (leave) => {
    setSelectedLeave(leave)
    setActionData({
      status: leave.status === 'Pending' ? 'Approved' : leave.status,
      remarks: leave.remarks || ''
    })
    setShowActionModal(true)
  }

  const handleAction = async () => {
    if (!actionData.status || actionData.status === selectedLeave.status) {
      alert('Please select a different status')
      return
    }

    try {
      setActionLoading(true)
      await managerAPI.actionLeave(selectedLeave.id, {
        status: actionData.status,
        remarks: actionData.remarks || null
      })
      setShowActionModal(false)
      setSelectedLeave(null)
      setActionData({ status: '', remarks: '' })
      fetchLeaves()
      fetchStats()
    } catch (error) {
      const message = error.response?.data?.message || error.response?.data?.detail || 'Failed to update leave request'
      alert(message)
    } finally {
      setActionLoading(false)
    }
  }

  const getStatusColor = (status) => {
    switch (status) {
      case 'Approved':
        return 'bg-green-100 text-green-800 border-green-200'
      case 'Rejected':
        return 'bg-red-100 text-red-800 border-red-200'
      case 'Pending':
        return 'bg-yellow-100 text-yellow-800 border-yellow-200'
      default:
        return 'bg-gray-100 text-gray-800 border-gray-200'
    }
  }

  const getStatusIcon = (status) => {
    switch (status) {
      case 'Approved':
        return (
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        )
      case 'Rejected':
        return (
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        )
      case 'Pending':
        return (
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        )
      default:
        return null
    }
  }

  const getLeaveTypeColor = (type) => {
    return type === 'Casual'
      ? 'bg-blue-100 text-blue-800'
      : 'bg-purple-100 text-purple-800'
  }

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
    })
  }

  const calculateDays = (startDate, endDate) => {
    const start = new Date(startDate)
    const end = new Date(endDate)
    const diffTime = Math.abs(end - start)
    const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24)) + 1
    return diffDays
  }

  const totalPages = Math.ceil(pagination.total / pagination.limit)
  const currentPage = Math.floor(pagination.skip / pagination.limit) + 1

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Header */}
        <div>
          <h1 className="text-3xl font-bold text-gray-900 mb-2">Leave Requests Management</h1>
          <p className="text-gray-600">Review and manage all employee leave requests</p>
        </div>

        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-100">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Total Requests</p>
                <p className="text-3xl font-bold text-gray-900 mt-2">{stats.total}</p>
              </div>
              <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center">
                <svg className="w-6 h-6 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                </svg>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-100">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Pending</p>
                <p className="text-3xl font-bold text-yellow-600 mt-2">{stats.pending}</p>
              </div>
              <div className="w-12 h-12 bg-yellow-100 rounded-lg flex items-center justify-center">
                <svg className="w-6 h-6 text-yellow-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-100">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Approved</p>
                <p className="text-3xl font-bold text-green-600 mt-2">{stats.approved}</p>
              </div>
              <div className="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center">
                <svg className="w-6 h-6 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-100">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Rejected</p>
                <p className="text-3xl font-bold text-red-600 mt-2">{stats.rejected}</p>
              </div>
              <div className="w-12 h-12 bg-red-100 rounded-lg flex items-center justify-center">
                <svg className="w-6 h-6 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
            </div>
          </div>
        </div>

        {/* Filters */}
        <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-100">
          <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-6 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Status</label>
              <select
                value={filters.status}
                onChange={(e) => handleFilterChange('status', e.target.value)}
                className="w-full px-4 py-2 rounded-lg border border-gray-300 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              >
                <option value="">All Status</option>
                <option value="Pending">Pending</option>
                <option value="Approved">Approved</option>
                <option value="Rejected">Rejected</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Leave Type</label>
              <select
                value={filters.leave_type}
                onChange={(e) => handleFilterChange('leave_type', e.target.value)}
                className="w-full px-4 py-2 rounded-lg border border-gray-300 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              >
                <option value="">All Types</option>
                <option value="Casual">Casual</option>
                <option value="Sick">Sick</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Employee ID</label>
              <input
                type="text"
                value={filters.employee_id}
                onChange={(e) => handleFilterChange('employee_id', e.target.value)}
                placeholder="Search..."
                className="w-full px-4 py-2 rounded-lg border border-gray-300 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Department</label>
              <select
                value={filters.department}
                onChange={(e) => handleFilterChange('department', e.target.value)}
                className="w-full px-4 py-2 rounded-lg border border-gray-300 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              >
                <option value="">All Departments</option>
                <option value="HR">HR</option>
                <option value="IT">IT</option>
                <option value="Finance">Finance</option>
                <option value="Marketing">Marketing</option>
                <option value="Sales">Sales</option>
                <option value="Operations">Operations</option>
                <option value="Engineering">Engineering</option>
                <option value="Other">Other</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Start Date</label>
              <input
                type="date"
                value={filters.start_date}
                onChange={(e) => handleFilterChange('start_date', e.target.value)}
                className="w-full px-4 py-2 rounded-lg border border-gray-300 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">End Date</label>
              <input
                type="date"
                value={filters.end_date}
                onChange={(e) => handleFilterChange('end_date', e.target.value)}
                className="w-full px-4 py-2 rounded-lg border border-gray-300 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              />
            </div>
          </div>
          <div className="mt-4">
            <button
              onClick={clearFilters}
              className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 transition duration-200"
            >
              Clear Filters
            </button>
          </div>
        </div>

        {/* Results Count */}
        <div className="text-sm text-gray-600">
          Showing {leaves.length} of {pagination.total} leave request(s)
        </div>

        {/* Leaves List */}
        {loading ? (
          <div className="bg-white rounded-xl shadow-lg p-12 border border-gray-100 text-center">
            <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
            <p className="mt-4 text-gray-600">Loading leave requests...</p>
          </div>
        ) : leaves.length === 0 ? (
          <div className="bg-white rounded-xl shadow-lg p-12 border border-gray-100 text-center">
            <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
            </svg>
            <h3 className="mt-4 text-lg font-medium text-gray-900">No leave requests found</h3>
            <p className="mt-2 text-sm text-gray-500">Try adjusting your filters</p>
          </div>
        ) : (
          <div className="space-y-4">
            {leaves.map((leave) => (
              <div
                key={leave.id}
                className="bg-white rounded-xl shadow-lg p-6 border border-gray-100 hover:shadow-xl transition duration-200"
              >
                <div className="flex flex-col lg:flex-row lg:items-start lg:justify-between gap-4">
                  <div className="flex-1">
                    <div className="flex items-center space-x-3 mb-3">
                      <span className={`px-3 py-1 text-xs font-semibold rounded-full ${getLeaveTypeColor(leave.leave_type)}`}>
                        {leave.leave_type}
                      </span>
                      <span className={`px-3 py-1 text-xs font-semibold rounded-full border flex items-center space-x-1 ${getStatusColor(leave.status)}`}>
                        {getStatusIcon(leave.status)}
                        <span>{leave.status}</span>
                      </span>
                      <span className="text-sm text-gray-500">Request ID: #{leave.id}</span>
                    </div>
                    <div className="space-y-2">
                      {leave.user && (
                        <div className="flex items-center text-sm text-gray-600 mb-2">
                          <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                          </svg>
                          <span className="font-medium">{leave.user.first_name} {leave.user.last_name}</span>
                          <span className="mx-2 text-gray-400">•</span>
                          <span className="text-gray-500">{leave.user.employee_id}</span>
                          <span className="mx-2 text-gray-400">•</span>
                          <span className="text-gray-500">{leave.user.department}</span>
                        </div>
                      )}
                      <div className="flex items-center text-sm text-gray-600">
                        <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
                        </svg>
                        <span className="font-medium">{formatDate(leave.start_date)}</span>
                        <span className="mx-2">-</span>
                        <span className="font-medium">{formatDate(leave.end_date)}</span>
                        <span className="ml-2 text-gray-500">({calculateDays(leave.start_date, leave.end_date)} day(s))</span>
                      </div>
                      <p className="text-sm text-gray-700">{leave.reason}</p>
                      {leave.remarks && (
                        <div className="mt-2 p-3 bg-gray-50 rounded-lg">
                          <p className="text-xs font-medium text-gray-600 mb-1">Manager Remarks:</p>
                          <p className="text-sm text-gray-700">{leave.remarks}</p>
                        </div>
                      )}
                      <p className="text-xs text-gray-500">
                        Applied on {formatDate(leave.created_at)}
                      </p>
                    </div>
                  </div>
                  {leave.status === 'Pending' && (
                    <div className="flex gap-2">
                      <button
                        onClick={() => openActionModal(leave)}
                        className="px-4 py-2 text-sm font-medium text-white bg-gradient-to-r from-blue-600 to-indigo-600 rounded-lg hover:shadow-lg transform hover:-translate-y-0.5 transition-all duration-200"
                      >
                        Take Action
                      </button>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="flex items-center justify-between bg-white rounded-xl shadow-lg p-4 border border-gray-100">
            <div className="text-sm text-gray-600">
              Page {currentPage} of {totalPages}
            </div>
            <div className="flex space-x-2">
              <button
                onClick={() => setPagination(prev => ({ ...prev, skip: Math.max(0, prev.skip - prev.limit) }))}
                disabled={pagination.skip === 0}
                className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed transition duration-200"
              >
                Previous
              </button>
              <button
                onClick={() => setPagination(prev => ({ ...prev, skip: prev.skip + prev.limit }))}
                disabled={pagination.skip + pagination.limit >= pagination.total}
                className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed transition duration-200"
              >
                Next
              </button>
            </div>
          </div>
        )}

        {/* Action Modal */}
        {showActionModal && selectedLeave && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
            <div className="bg-white rounded-2xl shadow-xl max-w-md w-full p-6">
              <h2 className="text-2xl font-bold text-gray-900 mb-4">Take Action on Leave Request</h2>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Action</label>
                  <select
                    value={actionData.status}
                    onChange={(e) => setActionData(prev => ({ ...prev, status: e.target.value }))}
                    className="w-full px-4 py-2 rounded-lg border border-gray-300 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  >
                    <option value="Approved">Approve</option>
                    <option value="Rejected">Reject</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Remarks (Optional)</label>
                  <textarea
                    value={actionData.remarks}
                    onChange={(e) => setActionData(prev => ({ ...prev, remarks: e.target.value }))}
                    rows={4}
                    className="w-full px-4 py-2 rounded-lg border border-gray-300 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 resize-none"
                    placeholder="Add any remarks or comments..."
                  />
                </div>
              </div>

              <div className="flex space-x-4 mt-6">
                <button
                  onClick={handleAction}
                  disabled={actionLoading || actionData.status === selectedLeave.status}
                  className="flex-1 bg-gradient-to-r from-blue-600 to-indigo-600 text-white py-3 px-4 rounded-lg font-semibold shadow-lg hover:shadow-xl transform hover:-translate-y-0.5 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none"
                >
                  {actionLoading ? 'Processing...' : 'Submit'}
                </button>
                <button
                  onClick={() => {
                    setShowActionModal(false)
                    setSelectedLeave(null)
                    setActionData({ status: '', remarks: '' })
                  }}
                  className="px-6 py-3 bg-white text-gray-700 font-semibold rounded-lg border-2 border-gray-300 hover:border-gray-400 transition duration-200"
                >
                  Cancel
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </DashboardLayout>
  )
}

export default ManagerRequests

