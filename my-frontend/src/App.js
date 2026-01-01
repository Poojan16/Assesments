import { Routes, Route, Router } from 'react-router-dom';
import React, { useEffect, useState } from 'react';
import { io } from 'socket.io-client';
import SignupPage from './components/signup';
// import LoginPage from './components/login';
import ProjectManagerHome from './components/studentForm';
import TaskExecutorHome from './components/password';
import TaskViewPage from './components/task-view';
import UpdateTaskPage from './components/head_teacher';
import DashboardTE from './components/reviewReportAudit';
import Dashboard from './components/Dashboard';
import CreateTaskForm from './components/teacherForm';
import LoginPage from './components/login';
import UpdateTaskPage1 from './components/update-task1';
import AuditPage from './components/audit';
import ProtectedRoute from './components/privateRoutes';
import ForgotPasswordForm from './components/resetPassword';
import NewPasswordForm from './components/setPassword';
import ReviewReportAudit from './components/reviewReportAudit';
import HeadTeacher from './components/head_teacher';
import TeacherMapping from './components/teacherMap';
import TeacherDashboard from './components/headTeacherdashboard';
import AddTeacherForm from './components/teacherForm';
import AddStudentForm from './components/studentForm';
import AdminDashboard2 from './components/AdminHome';
import StudentReportCard from './components/finalReport';
import TeacherStudentManager from './components/schoolPage';
import ClassSummaryReport from './components/summaryReport';
import UserAudit from './components/usersAudit';
import SubjectTeacherDashboard from './components/task-view';
import TopSchoolsDashboard from './components/admin_dashboard_prac';
import SubjectReportCard from './components/subjectReport';
import ReportConfirmationSystem from './components/admin_dashboard_prac';
import MyPdfViewer from './components/pdfViewer';
import ClassTeacherDashboard from './components/schoolPage';
import { LoadingProvider } from './loadingContext';
import GlobalLoader from './components/GlobalLoader';
import UnauthorizedPage from './components/Unauthorised';
import ParentComponent from './components/update-task1';
import EducationDashboard from './components/admin_dashboard_prac';
import SchoolPerformanceScatter from './components/update-task1';
import AddSchoolForm from './components/schoolAdd';
import { useDispatch } from 'react-redux';
import { checkAndInitializeAuth, validateSessionOnLoad } from './authSlice';

function App() {
  const dispatch = useDispatch();

  useEffect(() => {
    dispatch(validateSessionOnLoad());
  }, [dispatch]);

  return (
 
      <LoadingProvider>
        <GlobalLoader />
        <Routes>
          <Route path="/" element={<LoginPage />} />
          <Route path="/unauthorized" element={<UnauthorizedPage />} />
          <Route path="/signup" element={<SignupPage />} />
          <Route path="/forgot-password" element={<ForgotPasswordForm />} />
          <Route path="/set-password/:id" element={<NewPasswordForm />} />
          <Route path="/prac" element={<EducationDashboard />} />
          <Route path="/prac1" element={<SchoolPerformanceScatter />} />
          <Route path="/reviewAudit" element={<ReviewReportAudit />} />
          <Route path="/audit" element={<AuditPage />} />
          <Route path="/forgot-password/:email/:token" element={<ForgotPasswordForm />} />
          <Route path="/head-dashboard" element={<TeacherDashboard />} />
          <Route path="/subject-report/:id" element={<SubjectReportCard />} />
          <Route path="/user-audit" element={<UserAudit />} />
          <Route path="/subject-dashboard" element={<SubjectTeacherDashboard />} />
          <Route path="/set-password/:token" element={<NewPasswordForm />} />
          <Route path="/add-teacher" element={<AddTeacherForm />} />

              <Route path="/admin" element={<AdminDashboard2 />} />
          <Route element={<ProtectedRoute allowedRoles={[1]}/>}>
              <Route path="/add-school" element={<AddSchoolForm />} />

          </Route>

          <Route element={<ProtectedRoute allowedRoles={[2]}/>}>
              <Route path="/head-teacher" element={<HeadTeacher />} />
              <Route path="/edit-student/:id" element={<AddStudentForm />} />
              <Route path="/final-report/:id" element={<StudentReportCard />} />
              <Route path="/summary" element={<ClassSummaryReport />} />
              <Route path="/map-teacher" element={<TeacherMapping />} />
              <Route path="/add-student" element={<AddStudentForm />} />
          </Route>

          <Route element={<ProtectedRoute allowedRoles={[3]} />}>
              <Route path="/class-teacher" element={<ClassTeacherDashboard />} />
          </Route>

          {/* Add more routes and protected routes as needed */}
        </Routes>
      </LoadingProvider>
    
  );
}

export default App;
