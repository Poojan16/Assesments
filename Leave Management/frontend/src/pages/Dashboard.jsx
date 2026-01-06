import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'
import { leaveAPI, managerAPI } from '../services/api'
import DashboardLayout from '../layouts/DashboardLayout'

const Dashboard = () => {
  const { user } = useAuth()
  const isManager = user?.role === 'Manager'
  const [stats, setStats] = useState({
    total: 0,
    pending: 0,
    approved: 0,
    rejected: 0,
  })
  const [recentLeaves, setRecentLeaves] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchDashboardData()
  }, [])

  const fetchDashboardData = async () => {
    try {
      setLoading(true)
      if (isManager) {
        // Manager dashboard
        const statsResponse = await managerAPI.getStats()
        if (statsResponse.data.success) {
          setStats(statsResponse.data.data)
        }
        const leavesResponse = await managerAPI.getAllLeaves({ limit: 5 })
        setRecentLeaves(leavesResponse.data.data || [])
      } else {
        // Employee dashboard
        const response = await leaveAPI.getMyLeaves({ limit: 5 })
        const leaves = response.data.data || []
        
        setRecentLeaves(leaves)
        
        // Calculate stats
        const allLeavesResponse = await leaveAPI.getMyLeaves({ limit: 1000 })
        const allLeaves = allLeavesResponse.data.data || []
        
        setStats({
          total: allLeaves.length,
          pending: allLeaves.filter(l => l.status === 'Pending').length,
          approved: allLeaves.filter(l => l.status === 'Approved').length,
          rejected: allLeaves.filter(l => l.status === 'Rejected').length,
        })
      }
    } catch (error) {
      console.error('Error fetching dashboard data:', error)
    } finally {
      setLoading(false)
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

  const getLeaveTypeColor = (type) => {
    return type === 'Casual'
      ? 'bg-blue-100 text-blue-800'
      : 'bg-purple-100 text-purple-800'
  }

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Welcome Section */}
        <div className="bg-gradient-to-r from-blue-600 to-indigo-600 rounded-2xl shadow-xl p-8 text-white">
          <h1 className="text-3xl font-bold mb-2">
            Welcome back, {user?.first_name}! 👋
          </h1>
          <p className="text-blue-100">
            Manage your leave requests and track your time off from here.
          </p>
        </div>

        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-100">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Total Leaves</p>
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

        {/* Quick Actions */}
        {!isManager && (
          <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-100">
            <h2 className="text-xl font-bold text-gray-900 mb-4">Quick Actions</h2>
            <div className="flex flex-wrap gap-4">
              <Link
                to="/apply-leave"
                className="inline-flex items-center px-6 py-3 bg-gradient-to-r from-blue-600 to-indigo-600 text-white font-semibold rounded-lg shadow-lg hover:shadow-xl transform hover:-translate-y-0.5 transition-all duration-200"
              >
                <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                </svg>
                Apply for Leave
              </Link>
              <Link
                to="/my-leaves"
                className="inline-flex items-center px-6 py-3 bg-white text-gray-700 font-semibold rounded-lg border-2 border-gray-300 hover:border-blue-500 hover:text-blue-600 transition duration-200"
              >
                <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                </svg>
                View All Leaves
              </Link>
            </div>
          </div>
        )}
        {isManager && (
          <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-100">
            <h2 className="text-xl font-bold text-gray-900 mb-4">Quick Actions</h2>
            <div className="flex flex-wrap gap-4">
              <Link
                to="/manager/requests"
                className="inline-flex items-center px-6 py-3 bg-gradient-to-r from-blue-600 to-indigo-600 text-white font-semibold rounded-lg shadow-lg hover:shadow-xl transform hover:-translate-y-0.5 transition-all duration-200"
              >
                <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-3 7h3m-3 4h3m-6-4h.01M9 16h.01" />
                </svg>
                Manage Leave Requests
              </Link>
            </div>
          </div>
        )}

        {/* Recent Leaves */}
        <div className="bg-white rounded-xl shadow-lg border border-gray-100">
          <div className="p-6 border-b border-gray-200">
            <h2 className="text-xl font-bold text-gray-900">Recent Leave Requests</h2>
          </div>
          <div className="p-6">
            {loading ? (
              <div className="text-center py-8">
                <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
                <p className="mt-2 text-gray-600">Loading...</p>
              </div>
            ) : recentLeaves.length === 0 ? (
              <div className="text-center py-8">
                <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                </svg>
                <p className="mt-2 text-gray-600">No leave requests yet</p>
                <Link
                  to="/apply-leave"
                  className="mt-4 inline-block text-blue-600 hover:text-blue-700 font-medium"
                >
                  Apply for your first leave →
                </Link>
              </div>
            ) : (
              <div className="space-y-4">
                {recentLeaves.map((leave) => (
                  <div
                    key={leave.id}
                    className="flex items-center justify-between p-4 border border-gray-200 rounded-lg hover:bg-gray-50 transition duration-200"
                  >
                    <div className="flex-1">
                      <div className="flex items-center space-x-3">
                        <span className={`px-3 py-1 text-xs font-semibold rounded-full ${getLeaveTypeColor(leave.leave_type)}`}>
                          {leave.leave_type}
                        </span>
                        <span className={`px-3 py-1 text-xs font-semibold rounded-full border ${getStatusColor(leave.status)}`}>
                          {leave.status}
                        </span>
                      </div>
                      <p className="mt-2 text-sm text-gray-600">
                        {new Date(leave.start_date).toLocaleDateString()} - {new Date(leave.end_date).toLocaleDateString()}
                      </p>
                      <p className="mt-1 text-sm text-gray-500 line-clamp-1">{leave.reason}</p>
                    </div>
                  </div>
                ))}
                <Link
                  to="/my-leaves"
                  className="block text-center mt-4 text-blue-600 hover:text-blue-700 font-medium"
                >
                  View all leaves →
                </Link>
              </div>
            )}
          </div>
        </div>
      </div>
    </DashboardLayout>
  )
}

export default Dashboard

