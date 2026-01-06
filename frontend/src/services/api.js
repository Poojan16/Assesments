import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

// Create axios instance
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Request interceptor to add auth token
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('access_token')
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// Response interceptor to handle errors
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Unauthorized - clear token and redirect to login
      localStorage.removeItem('access_token')
      localStorage.removeItem('user')
      window.location.href = '/login'
    }
    return Promise.reject(error)
  }
)

// Auth API
export const authAPI = {
  signup: (data) => api.post('/api/auth/signup', data),
  login: (data) => api.post('/api/auth/login', data),
}

// Leave API
export const leaveAPI = {
  applyLeave: (data) => api.post('/api/leaves/apply', data),
  getMyLeaves: (params = {}) => {
    const queryParams = new URLSearchParams()
    if (params.status) queryParams.append('status', params.status)
    if (params.leave_type) queryParams.append('leave_type', params.leave_type)
    if (params.start_date) queryParams.append('start_date', params.start_date)
    if (params.end_date) queryParams.append('end_date', params.end_date)
    if (params.skip) queryParams.append('skip', params.skip)
    if (params.limit) queryParams.append('limit', params.limit)
    
    const queryString = queryParams.toString()
    return api.get(`/api/leaves/my-leaves${queryString ? `?${queryString}` : ''}`)
  },
  getLeaveById: (leaveId) => api.get(`/api/leaves/${leaveId}`),
}

// Manager API
export const managerAPI = {
  getAllLeaves: (params = {}) => {
    const queryParams = new URLSearchParams()
    if (params.status) queryParams.append('status', params.status)
    if (params.leave_type) queryParams.append('leave_type', params.leave_type)
    if (params.start_date) queryParams.append('start_date', params.start_date)
    if (params.end_date) queryParams.append('end_date', params.end_date)
    if (params.employee_id) queryParams.append('employee_id', params.employee_id)
    if (params.department) queryParams.append('department', params.department)
    if (params.skip) queryParams.append('skip', params.skip)
    if (params.limit) queryParams.append('limit', params.limit)
    
    const queryString = queryParams.toString()
    return api.get(`/api/manager/leaves${queryString ? `?${queryString}` : ''}`)
  },
  getLeaveById: (leaveId) => api.get(`/api/manager/leaves/${leaveId}`),
  actionLeave: (leaveId, data) => api.patch(`/api/manager/leaves/${leaveId}/action`, data),
  getStats: () => api.get('/api/manager/stats'),
}

export default api

