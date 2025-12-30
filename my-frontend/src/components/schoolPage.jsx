// import React, { useState, useMemo, useEffect, useCallback, useRef } from 'react';
// import { 
//   Search, Download, X, Plus, Send, FileText, ZoomIn, ZoomOut, 
//   CheckCircle, XCircle, MessageSquare, Check, Upload, LogOut,
//   BarChart3, TrendingUp, Users, GraduationCap, BookOpen, Clock,
//   Calendar, Award, Star, AlertCircle, Bell, Filter, ChevronRight,
//   ChevronDown, PieChart, Target, Bookmark, UserCheck, ShieldCheck,
//   MessageCircle, ThumbsUp, Eye, Edit2, Trash2, MoreVertical,
//   Home, CheckSquare, Square, PenTool, Signature, SendHorizonal,
//   ChevronLeft
// } from 'lucide-react';
// import { useDispatch, useSelector } from 'react-redux';
// import { useNavigate } from 'react-router-dom';
// import ExcelJS from 'exceljs';
// import Loader from './loader';
// import MyPdfViewer from './pdfViewer';
// import SubjectReportCard from './subjectReport';
// import { initializeAuth, logout } from '../authSlice';
// import BatchStudentReportCard from './subjectReport';

// const ClassTeacherDashboard = () => {
//   const { user } = useSelector((state) => state.auth);
//   console.log('User:', user);
//   const navigate = useNavigate();

//   const dispatch = useDispatch();

//   useEffect(() => {
//     // Initialize auth from localStorage on app load
//     dispatch(initializeAuth());
//   }, [dispatch]);

//   // State management
//   const [subjects, setSubjects] = useState([]);
//   const [classes, setClasses] = useState([]);
//   const [students, setStudents] = useState([]);
//   const [teacherSubjects, setTeacherSubjects] = useState([]);
//   const [studentScoresData, setStudentScoresData] = useState([]);
//   const [teacherClasses, setTeacherClasses] = useState([]);
//   const [loading, setLoading] = useState(true);
//   const [activeTab, setActiveTab] = useState('dashboard');
//   const [searchQuery, setSearchQuery] = useState('');
//   const [selectedStatus, setSelectedStatus] = useState('All');
//   const [selectedClass, setSelectedClass] = useState('All');
//   const [selectedSubject, setSelectedSubject] = useState('All');
//   const [showScoreModal, setShowScoreModal] = useState(false);
//   const [selectedStudent, setSelectedStudent] = useState(null);
//   const [selectedScoreForEdit, setSelectedScoreForEdit] = useState(null);
//   const [scoreData, setScoreData] = useState({ subject: '', score: '', student_id: '' });
//   const [successMessage, setSuccessMessage] = useState('');
//   const [errorMessage, setErrorMessage] = useState('');
//   const [showUploadModal, setShowUploadModal] = useState(false);
//   const [uploadedReport, setUploadedReport] = useState(null);
//   const [popUpMsg, setPopupMsg] = useState('');
//   const [reports, setReports] = useState([]);
//   const [teacher, setTeacher] = useState({});
//   const [mapTeacher, setMapTeacher] = useState([]);
//   const [quickStats, setQuickStats] = useState({
//     totalStudents: 0,
//     averageScore: 0,
//     topPerformers: 0,
//     pendingScores: 0,
//     recentSubmissions: 0
//   });
//   const [performanceTrend, setPerformanceTrend] = useState([]);
//   const [grades, setGrades] = useState([]);
//   const [school, setSchool] = useState({});
  
//   // New states for signature and bulk operations
//   const [signatureSubject, setSignatureSubject] = useState([]);
//   const [uploadedSignature, setUploadedSignature] = useState(null);
//   const [signatures, setSignatures] = useState({});
//   const [selectedStudents, setSelectedStudents] = useState(new Set());
//   const [selectAll, setSelectAll] = useState(false);
//   const [showSendReportModal, setShowSendReportModal] = useState(false);
//   const [sendingReports, setSendingReports] = useState(false);
//   const [comments, setComments] = useState([]);
//   const [review, setReviews] = useState([]);



//     // Signature states
  
//   const [selectedReport, setSelectedReport] = useState(null);
//   const [zoomLevel, setZoomLevel] = useState(100);
//   const [comment, setComment] = useState('');
//   const [showSuccess, setShowSuccess] = useState(false);

//   // Fetch teacher data
//   useEffect(() => {
//     const fetchTeacher = async () => {
//       try {
//         if (!user?.userEmail) return;
        
//         const res = await fetch(
//           `http://127.0.0.1:8000/teachers/email?email=${user.userEmail}`
//         );
        
//         if (!res.ok) {
//           throw new Error(`HTTP error! status: ${res.status}`);
//         }
        
//         const data = await res.json();
//         setTeacher(data?.data);
//       } catch (err) {
//         console.error("Error fetching teacher:", err);
//       }
//     };
  
//     fetchTeacher();
//   }, [user]);

//   // Fetch all data after teacher is loaded
//   useEffect(() => {
//     const fetchData = async () => {
//       if (!teacher?.teacherId) return;
      
//       setLoading(true);
//       try {
//         const teacherId = Number(teacher.teacherId);
        
//         // Fetch all data in parallel
//         const [
//           studentResponse, 
//           subjectResponse, 
//           classResponse, 
//           studentScoreResponse,
//           workFlowResponse,
//           MapResponse,
//           teacherSign,
//           gradesresponse,
//           schoolResponse
//         ] = await Promise.all([
//           fetch('http://127.0.0.1:8000/students/'),
//           fetch('http://127.0.0.1:8000/subjects/'),
//           fetch('http://127.0.0.1:8000/classes/'),
//           fetch(`http://127.0.0.1:8000/students/score`),
//           fetch(`http://127.0.0.1:8000/teachers/getWorkFlow?teacherId=${teacherId}`),
//           fetch(`http://127.0.0.1:8000/teachers/class_and_subjects?teacherId=${teacherId}`),
//           fetch(`http://127.0.0.1:8000/teachers/teacherSign/`),
//           fetch(`http://127.0.0.1:8000/grades/`),
//           fetch(`http://127.0.0.1:8000/admin/schools/${teacher?.schoolId}`)

//         ]);

//         // Check all responses
//         if (!studentResponse.ok) throw new Error('Failed to fetch students');
//         if (!subjectResponse.ok) throw new Error('Failed to fetch subjects');
//         if (!classResponse.ok) throw new Error('Failed to fetch classes');
//         if (!studentScoreResponse.ok) throw new Error('Failed to fetch scores');
//         if (!workFlowResponse.ok) throw new Error('Failed to fetch workflow');
//         if (!MapResponse.ok) throw new Error('Failed to fetch teacher mappings');
//         if (!teacherSign.ok) throw new Error('Failed to fetch teacher signature');
//         if (!gradesresponse.ok) throw new Error('Failed to fetch grades');
//         if (!schoolResponse.ok) throw new Error('Failed to fetch school');

//         const [
//           studentData, 
//           subjectData, 
//           classData, 
//           studentScores,
//           workflowData,
//           mapData,
//           teacherSignData,
//           grade,
//           school
//         ] = await Promise.all([
//           studentResponse.json(),
//           subjectResponse.json(),
//           classResponse.json(),
//           studentScoreResponse.json(),
//           workFlowResponse.json(),
//           MapResponse.json(),
//           teacherSign.json(),
//           gradesresponse.json(),
//           schoolResponse.json()
//         ]);

//         setSchool(school?.data || {});
//         setStudentScoresData(studentScores?.data || []);
//         setGrades(grade?.data || []);
//         setClasses(classData?.data || []);
//         setSignatures(teacherSignData?.data || {});
//         setMapTeacher(mapData?.data || []);


//         // Filter students by school
//         const filteredStudents = (studentData?.data || []).filter(student => 
//           student.schoolId === teacher.schoolId
//         );

//         const filteredSubjects = (subjectData?.data || []).filter(subject => {
//           const schoolClass = (classData?.data || []).find(cls => cls.classId === subject.classId);
//           return schoolClass?.schoolId === teacher.schoolId;
//         })
      
//         setStudents(filteredStudents || []);
//         setSubjects(filteredSubjects || []);

//         try {
//           const params = new URLSearchParams();
      
//           (workflowData?.data).forEach(r => {
//             params.append("reportIds", r.reportId);
//           });
      
//           const response = await fetch(
//             `http://127.0.0.1:8000/teachers/getWorkFlow/ids?${params.toString()}`
//           );
      
//           const data = await response.json();
//           console.log(data);
//           setReviews(data?.data || []);
//         } catch (error) {
//           console.error("Error fetching reviews:", error);
//         }

//         // Extract unique subjects and classes from map data
//         const uniqueSubjectIds = [...new Set((mapData?.data || []).map(item => item.subjectId))];
//         const uniqueClassIds = [...new Set((mapData?.data || []).map(item => item.classId))];

//         const teacherSubjectsData = (subjectData?.data || []).filter(subject => 
//           uniqueSubjectIds.includes(subject.subjectId)
//         );
        
//         const teacherClassesData = (classData?.data || []).filter(cls => 
//           uniqueClassIds.includes(cls.classId)
//         );

//         setTeacherSubjects(teacherSubjectsData);
//         setTeacherClasses(teacherClassesData);

//         // Process workflow attachments
//         const processedWorkflow = (workflowData?.data || []).map(item => {
//           if (item.report) {
//             try {
//               const base64 = item.report.replace(/\s/g, "");
//               const byteCharacters = atob(base64);
//               const byteArray = new Uint8Array(byteCharacters.length);
//               for (let i = 0; i < byteCharacters.length; i++) {
//                 byteArray[i] = byteCharacters.charCodeAt(i);
//               }
//               const blob = new Blob([byteArray], { type: "application/pdf" });
//               return {
//                 ...item,
//                 report: URL.createObjectURL(blob)
//               };
//             } catch (error) {
//               console.error("Error processing report:", error);
//               return item;
//             }
//           }
//           return item;
//         });
        
//         setReports(processedWorkflow);

//         // Calculate quick stats
//         const totalStudents = filteredStudents.length;
//         const scores = studentScores?.data || [];
//         const avgScore = totalStudents > 0 && scores.length > 0
//           ? scores.reduce((sum, score) => sum + (score.score || 0), 0) / scores.length
//           : 0;

//         const topPerformers = filteredStudents.filter(student => {
//           const studentScores = scores.filter(s => s.studentId === student.studentId);
//           if (studentScores.length === 0) return false;
//           const avg = studentScores.reduce((sum, s) => sum + (s.score || 0), 0) / studentScores.length;
//           return avg >= 80;
//         }).length;

//         const pendingScores = Math.max(
//           0, 
//           filteredStudents.length * teacherSubjectsData.length - scores.length
//         );

//         setQuickStats({
//           totalStudents,
//           averageScore: Math.round(avgScore * 100) / 100,
//           topPerformers,
//           pendingScores,
//           recentSubmissions: processedWorkflow.length
//         });

//         // Calculate performance trends
//         const trends = subjects.map(subject => {
//           const subjectScores = scores.filter(s => s.subjectId === subject.subjectId);
//           const avg = subjectScores.length > 0
//             ? subjectScores.reduce((sum, s) => sum + (s.score || 0), 0) / subjectScores.length
//             : 0;
          
//           return {
//             subject: subject.subjectName,
//             averageScore: Math.round(avg),
//             totalStudents: filteredStudents.filter(s => 
//               (mapData?.data || []).some(m => m.classId === s.classId && m.subjectId === subject.subjectId)
//             ).length,
//             trend: avg >= 75 ? 'up' : avg >= 60 ? 'stable' : 'down'
//           };
//         });

//         setPerformanceTrend(trends);

//       } catch (error) {
//         console.error('Error fetching data:', error);
//         setErrorMessage('Failed to load data. Please try again.');
//         const timer = setTimeout(() => {
//           setErrorMessage(null);
//         }, 3000);
//       } finally {
//         setLoading(false);
//       }
//     };

//     fetchData();
//   }, [teacher]);

//   // Filter students
//   const filteredStudents = useMemo(() => {
//     if (!students.length) return [];
    
//     return students.filter(student => {
//       const matchesSearch = searchQuery === '' || 
//         student.studentName?.toLowerCase().includes(searchQuery.toLowerCase());
      
//       const matchesClass = selectedClass === 'All' || 
//         String(student.classId) === selectedClass;
      
//       // Check if student belongs to any of teacher's classes
//       const studentInTeacherClass = teacherClasses.some(cls => 
//         cls.classId === student.classId
//       );
      
//       const matchesSubject = selectedSubject === 'All' || 
//         teacherSubjects.some(subject => 
//           subject.subjectId === Number(selectedSubject)
//         );
      
//       const matchesStatus = selectedStatus === 'All' || 
//         String(student.active) === selectedStatus;
      
//       return matchesSearch && matchesClass && matchesSubject && matchesStatus && studentInTeacherClass;
//     });
//   }, [students, searchQuery, selectedClass, selectedSubject, selectedStatus, teacherSubjects, teacherClasses]);

//   // Handle score submission or update
//   const handleSaveScore = async () => {
//     if (!selectedStudent || !scoreData.subject || !scoreData.score) {
//       setErrorMessage('Please select a subject and enter a score');
//       return;
//     }

//     const isEdit = !!selectedScoreForEdit;
//     const url = isEdit 
//       ? `http://127.0.0.1:8000/students/score/${selectedScoreForEdit}`
//       : `http://127.0.0.1:8000/students/score`;
    
//     const method = isEdit ? 'PUT' : 'POST';

//     const formData = new FormData();
//     if (isEdit) {
//       formData.append('score', scoreData.score);
//     } else {
//       formData.append('studentId', selectedStudent.studentId);
//       formData.append('subjectId', scoreData.subject);
//       formData.append('score', scoreData.score);
//       formData.append('teacherId', teacher.teacherId);
//     }

//     try {
//       const response = await fetch(url, {
//         method,
//         body: formData,
//       });
      
//       if (!response.ok) {
//         throw new Error(`HTTP error! status: ${response.status}`);
//       }
      
//       const data = await response.json();
      
//       if (data?.status_code === 200) {
//         setSuccessMessage(`Score ${isEdit ? 'updated' : 'added'} successfully`);
//         setTimeout(() => {
//           setShowScoreModal(false);
//           setSuccessMessage('');
//           setScoreData({ subject: '', score: '', student_id: '' });
//           setSelectedStudent(null);
//           setSelectedScoreForEdit(null);
//           window.location.reload();
//         }, 2000);
//       } else {
//         setErrorMessage(`Score ${isEdit ? 'update' : 'addition'} failed`);
//       }
//     } catch (error) {
//       console.error('Error saving score:', error);
//       setErrorMessage(`Failed to ${isEdit ? 'update' : 'save'} score. Please try again.`);
//     }
//   };

//   // Handle add/edit score
//   const handleAddEditScore = (student, subjectId, existingScore = null) => {
//     setSelectedStudent(student);
//     setSelectedScoreForEdit(existingScore?.scoreId || null);
    
//     if (existingScore) {
//       setScoreData({ 
//         subject: existingScore.subjectId, 
//         score: existingScore.score,
//         student_id: student.studentId 
//       });
//     } else {
//       setScoreData({ 
//         subject: subjectId, 
//         score: '', 
//         student_id: student.studentId 
//       });
//     }
    
//     setShowScoreModal(true);
//     setSuccessMessage('');
//     setErrorMessage('');
//   };

//   // Handle select/deselect all
//   const handleSelectAll = () => {
//     if (selectAll) {
//       setSelectedStudents(new Set());
//     } else {
//       const allIds = new Set(filteredStudents.map(student => String(student.studentId)));
//       setSelectedStudents(allIds);
//     }
//     setSelectAll(!selectAll);
//   };

//   // Handle individual student selection
//   const handleSelectStudent = (studentId) => {
//     const newSelected = new Set(selectedStudents);
//     if (newSelected.has(studentId)) {
//       newSelected.delete(studentId);
//     } else {
//       newSelected.add(studentId);
//     }
//     setSelectedStudents(newSelected);
    
//     // Update selectAll state
//     if (newSelected.size === filteredStudents.length) {
//       setSelectAll(true);
//     } else if (selectAll) {
//       setSelectAll(false);
//     }
//   };

//   // Export to Excel
//   const exportToExcel = async () => {
//     if (filteredStudents.length === 0) {
//       setErrorMessage('No data to export');
//       setTimeout(() => setErrorMessage(''), 3000);
//       return;
//     }

//     try {
//       const workbook = new ExcelJS.Workbook();
//       const worksheet = workbook.addWorksheet('Student Scores');

//       // Prepare headers
//       const headers = ['Roll ID', 'Student Name', 'Class', 'Status'];
//       const scoreHeaders = teacherSubjects.map(subject => 
//         `${subject.subjectName} Score`
//       );
//       const signatureHeaders = teacherSubjects.map(subject =>
//         `${subject.subjectName} Signature`
//       );
//       const allHeaders = [...headers, ...scoreHeaders, ...signatureHeaders];

//       // Set column widths and add headers
//       worksheet.columns = allHeaders.map(header => ({
//         header,
//         width: 25
//       }));

//       // Style headers
//       const headerRow = worksheet.getRow(1);
//       headerRow.eachCell((cell) => {
//         cell.font = { bold: true, color: { argb: 'FFFFFFFF' }, name: 'Arial', size: 12 };
//         cell.fill = {
//           type: 'pattern',
//           pattern: 'solid',
//           fgColor: { argb: '1A47C2' },
//         };
//         cell.alignment = { horizontal: 'center', vertical: 'middle' };
//         cell.border = {
//           top: { style: 'medium' },
//           left: { style: 'medium' },
//           bottom: { style: 'medium' },
//           right: { style: 'medium' }
//         };
//       });
      
//       // freezing header
//       worksheet.views = [
//         {
//           state: 'frozen',
//           xSplit: 0,
//           ySplit: 1, 
//           activeCell: 'A2',
//           showGridLines: true
//         }
//       ];


//       // Auto fit columns
//       worksheet.columns.forEach(column => {
//         let maxLength = 0;
//         column.eachCell({ includeEmpty: true }, cell => {
//           const columnLength = cell.value ? cell.value.toString().length : 10;
//           if (columnLength > maxLength) {
//             maxLength = columnLength;
//           }
//         });
//         column.width = maxLength < 10 ? 10 : maxLength + 2;
//       });

//       // Add data rows
//       filteredStudents.forEach((student, index) => {
//         const classInfo = classes.find(c => c.classId === student.classId);
        
//         const rowData = [
//           student.rollId || `CLS${student.classId}000${student.studentId}`,
//           student.studentName || 'N/A',
//           classInfo?.className || 'N/A',
//           student.active ? 'Active' : 'Inactive'
//         ];

//         // Add scores for each subject
//         teacherSubjects.forEach(subject => {
//           const score = studentScoresData.find(
//             s => s.studentId === student.studentId && s.subjectId === subject.subjectId
//           )?.score || 'N/A';
//           rowData.push(score);
//         });

//         // Add signature status for each subject
//         teacherSubjects.forEach(subject => {
//           const hasSignature = signatures[subject.subjectId] ? 'Signed' : 'Not Signed';
//           rowData.push(hasSignature);
//         });

//         const row = worksheet.addRow(rowData);
        
//         // Style data rows
//         row.eachCell((cell) => {
//           cell.font = { size: 11 };
//           cell.border = {
//             top: { style: 'thin' },
//             left: { style: 'thin' },
//             bottom: { style: 'thin' },
//             right: { style: 'thin' }
//           };
//           cell.alignment = { vertical: 'middle' };
//         });

//         // Alternate row colors
//         if (index % 2 === 0) {
//           row.eachCell((cell) => {
//             cell.fill = {
//               type: 'pattern',
//               pattern: 'solid',
//               fgColor: { argb: 'F3F4F6' }
//             };
//           });
//         }
//       });

//       // Auto-fit columns
//       worksheet.columns.forEach(column => {
//         let maxLength = 0;
//         column.eachCell({ includeEmpty: true }, cell => {
//           const cellLength = cell.value ? cell.value.toString().length : 10;
//           if (cellLength > maxLength) {
//             maxLength = cellLength;
//           }
//         });
//         column.width = Math.min(Math.max(maxLength + 2, 15), 30);
//       });

//       // Generate and download
//       const buffer = await workbook.xlsx.writeBuffer();
//       const blob = new Blob([buffer], { 
//         type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' 
//       });
//       const url = URL.createObjectURL(blob);
//       const a = document.createElement('a');
//       a.href = url;
//       a.download = `student_scores_${new Date().toISOString().split('T')[0]}.xlsx`;
//       document.body.appendChild(a);
//       a.click();
//       document.body.removeChild(a);
//       URL.revokeObjectURL(url);
//     } catch (error) {
//       console.error('Error exporting to Excel:', error);
//       setErrorMessage('Failed to expodatetimert data');
//       setTimeout(() => setErrorMessage(''), 3000);
//     }
//   };

//   // Cleanup URLs on unmount
//   useEffect(() => {
//     return () => {
//       reports.forEach(report => {
//         if (report.report && report.report.startsWith('blob:')) {
//           URL.revokeObjectURL(report.report);
//         }
//       });
//     };
//   }, [reports]);

//   const handleUploadReport = async () => {
//         if (!uploadedReport) {
//           setErrorMessage('Please select a file to upload');
//           return;
//         }
    
//         if (!teacher?.teacherId) {
//           setErrorMessage('Teacher information not available');
//           return;
//         }
    
//         const formData = new FormData();
//         formData.append('attachment', uploadedReport);
//         formData.append('teacherId', teacher.teacherId);
//         formData.append('comments', comment || 'Student subject wise Report');
    
//         try {
//           const response = await fetch('http://127.0.0.1:8000/teachers/uploadReport', {
//             method: 'POST',
//             body: formData
//           });
          
//           if (!response.ok) {
//             throw new Error(`HTTP error! status: ${response.status}`);
//           }
          
//           const data = await response.json();
//           setPopupMsg(data.message || 'Report uploaded successfully');
          
//           // Refresh reports after upload
//           const workflowResponse = await fetch(
//             `http://127.0.0.1:8000/teachers/getWorkFlow?teacherId=${teacher.teacherId}`
//           );
//           const workflowData = await workflowResponse.json();
          
//           (workflowData?.data).map(item => {
//             if (item.report) {
//               // Remove line breaks from base64 if any
//               const base64 = item.report.replace(/\s/g, "");
//               const byteCharacters = atob(base64);
//               console.log('byteCharacters',byteCharacters);
//               const byteArray = new Uint8Array(byteCharacters.length);
//               console.log('byteArray',byteArray);
//               for (let i = 0; i < byteCharacters.length; i++) {
//                 byteArray[i] = byteCharacters.charCodeAt(i);
//               }
//               const blob = new Blob([byteArray], { type: "application/pdf" });
//               console.log('blob',blob);
//               item.report = URL.createObjectURL(blob); // Replace with blob URL
//             }
//             return item;
//           });
          
//           setReports(workflowData);
          
//           setTimeout(() => {
//             setUploadedReport(null);
//             setShowUploadModal(false);
//             setPopupMsg('');
//             setComment('');
//           }, 3000);
          
//         } catch (error) {
//           console.error('Error uploading report:', error);
//           setErrorMessage('Failed to upload report');
//           setTimeout(() => setErrorMessage(''), 3000);
//         }
//       };
    
//       console.log(reports);
    
//       const handleReportClick = (report) => {
//         setSelectedReport(report);
//         console.log(report);
//         setZoomLevel(100);
//         setComment(report.comments || '');
//       };
    
//       const handleApprove = () => {
//         if (!selectedReport) return;
        
//         // Update local state
//         setReports(reports.map(r => 
//           r.reviewId === selectedReport.reviewId 
//             ? { ...r, iStatus: true, comments: comment || r.comments }
//             : r
//         ));
        
//         // Reset selection
//         setSelectedReport(null);
//         setComment('');
//       };
    
//       const handleReject = () => {
//         if (!selectedReport) return;
        
//         if (!comment.trim()) {
//           setErrorMessage('Please add comments before rejecting');
//           setTimeout(() => setErrorMessage(''), 3000);
//           return;
//         }
        
//         // Update local state
//         setReports(reports.map(r => 
//           r.reviewId === selectedReport.reviewId 
//             ? { ...r, iStatus: false, comments: comment }
//             : r
//         ));
        
//         // Reset selection
//         setSelectedReport(null);
//         setComment('');
//       };
    
//       const getStatusColor = (status) => {
//         if (status === true || status === 'true') {
//           return 'bg-green-100 text-green-800';
//         } else if (status === false || status === 'false') {
//           return 'bg-red-100 text-red-800';
//         }
//         return 'bg-yellow-100 text-yellow-800';
//       };
    
//       const getStatusText = (status) => {
//         if (status === true || status === 'true') {
//           return 'Approved';
//         } else if (status === false || status === 'false') {
//           return 'Rejected';
//         }
//         return 'Pending';
//       };

//       const getClassDisplayName = (classId) => {
//             const cls = classes.find(c => c.classId === classId);
//             if (!cls) return 'N/A';
            
//             const classStudents = students.filter(s => s.classId === classId);
//             return `${cls.className}`;
//           };


//        // Pagination calculations
//        const [currentPage, setCurrentPage] = useState(1);
//         const [itemsPerPage, setItemsPerPage] = useState(5);

//         const indexOfLastItem = currentPage * itemsPerPage;
//         const indexOfFirstItem = indexOfLastItem - itemsPerPage;
//         const currentItems = filteredStudents.slice(indexOfFirstItem, indexOfLastItem);
//         const totalPages = Math.ceil(filteredStudents.length / itemsPerPage);

//         const handlePageChange = (pageNumber) => {
//           if (pageNumber >= 1 && pageNumber <= totalPages) {
//             setCurrentPage(pageNumber);
//           }
//         };

//         const getPageNumbers = () => {
//           const pages = [];
//           const maxPagesToShow = 5;
        
//           if (totalPages <= maxPagesToShow) {
//             for (let i = 1; i <= totalPages; i++) pages.push(i);
//           } else {
//             pages.push(1);
//             if (currentPage > 3) pages.push('...');
//             const start = Math.max(2, currentPage - 1);
//             const end = Math.min(totalPages - 1, currentPage + 1);
//             for (let i = start; i <= end; i++) pages.push(i);
//             if (currentPage < totalPages - 2) pages.push('...');
//             pages.push(totalPages);
//           }
//           return pages;
//         };

//         // Reset to first page when filters change
//         useEffect(() => {
//           setCurrentPage(1);
//         }, [searchQuery, selectedClass, selectedSubject]);

//         const isAllSubjectsFilled = (studentId) => {
//           const studentScores = studentScoresData.filter(score => score.studentId === studentId);
//           console.log('studentScores',studentScores);
//           return studentScores.length === subjects.length;
//         };
      
//         // Check if selected students have all subjects filled
//         const canSendSelectedReports = () => {
//           if (selectedStudents.size === 0) return false;
          
//           for (const studentId of selectedStudents) {
//             if (!isAllSubjectsFilled(Number(studentId))) {
//               return false;
//             }
//           }
//           return true;
//         };

//         // Check if all subjects are signed by teacher
//           const isAllSubjectsSigned = (studentId) => {
//             return subjects.every(subject => {
//               return signatures.some(signature => 
//                 signature.subjectId === subject.subjectId && 
//                 signature.studentId === studentId
//               );
//             });
//           };

//           // Check if student is ready for confirmation (all filled AND all signed)
//           const isReadyForConfirmation = (studentId) => {
//             console.log(isAllSubjectsFilled(studentId), isAllSubjectsSigned(studentId));
//             return isAllSubjectsFilled(studentId) && isAllSubjectsSigned(studentId);
//           };


//           // Signature
//           const [signature, setSignature] = useState(null);
//           const [showSignatureModal, setShowSignatureModal] = useState(false);
//           const [isDrawing, setIsDrawing] = useState(false);
//           const canvasRef = useRef(null);
          
//         // Drawing functions
//           const startDrawing = (e) => {
//             const canvas = canvasRef.current;
//             const rect = canvas.getBoundingClientRect();
//             const ctx = canvas.getContext('2d');
//             ctx.beginPath();
//             ctx.moveTo(e.clientX - rect.left, e.clientY - rect.top);
//             setIsDrawing(true);
//           };

//           const draw = (e) => {
//             if (!isDrawing) return;
//             const canvas = canvasRef.current;
//             const rect = canvas.getBoundingClientRect();
//             const ctx = canvas.getContext('2d');
//             ctx.lineTo(e.clientX - rect.left, e.clientY - rect.top);
//             ctx.strokeStyle = '#1e40af';
//             ctx.lineWidth = 2;
//             ctx.lineCap = 'round';
//             ctx.stroke();
//           };

//           const stopDrawing = () => {
//             setIsDrawing(false);
//           };

//           const clearSignature = () => {
//             const canvas = canvasRef.current;
//             const ctx = canvas.getContext('2d');
//             ctx.clearRect(0, 0, canvas.width, canvas.height);
//           };

//           const dataURLToBlob = (dataURL) => {
//             const arr = dataURL.split(',');
//             const mime = arr[0].match(/:(.*?);/)[1];
//             const bstr = atob(arr[1]);
//             let n = bstr.length;
//             const u8arr = new Uint8Array(n);
          
//             while (n--) {
//               u8arr[n] = bstr.charCodeAt(n);
//             }
          
//             return new Blob([u8arr], { type: mime });
//           };
          

//           const saveSignature = async () => {
//             const canvas = canvasRef.current;
          
//             // Get Data URL from canvas
//             const signatureDataURL = canvas.toDataURL('image/png');
          
//             // Convert Data URL to Blob
//             const signatureBlob = dataURLToBlob(signatureDataURL);
          
//             // Optional: convert Blob to File (recommended)
//             const signatureFile = new File(
//               [signatureBlob],
//               'signature.png',
//               { type: 'image/png' }
//             );
          
//             const formData = new FormData();
//             formData.append('signature', signatureFile); // 👈 real image file
//             formData.append('subjectId', signatureSubject.subjectId);
//             formData.append('teacherId', teacher.teacherId);
//             formData.append('studentId', Array.from(selectedStudents));
//             formData.append(
//               'score',
//               studentScoresData.find(
//                 score => score.subjectId === signatureSubject.subjectId
//               )?.score || ''
//             );
          
//             const saveSign = await fetch('http://127.0.0.1:8000/teachers/teacherSign', {
//               method: 'POST',
//               body: formData,
//             });
          
//             if (saveSign.ok) {
//               const response = await saveSign.json();
//               console.log(response);
          
//               if (response?.status_code === 200) {
//                 setShowSignatureModal(false);
//                 setPopupMsg('Signature saved successfully');
          
//                 setTimeout(() => {
//                   setPopupMsg('');
//                   setShowSignatureModal(false);
//                   window.location.reload();
//                 }, 3000);
//               }
//             }
//           };
          
          

//           const handleSignatureUpload = (e) => {
//             const file = e.target.files[0];
//             if (file) {
//               const reader = new FileReader();
//               reader.onload = async () => {
//                 const signatureData = reader.result;
//                 const formData = new FormData();
//                 formData.append('signature', file);
//                 formData.append('subjectId', signatureSubject.subjectId);
//                 formData.append('teacherId', teacher.teacherId);
//                 formData.append('studentId', Array.from(selectedStudents));
//                 formData.append('score', studentScoresData.find(score => score.subjectId === signatureSubject.subjectId)?.score || '');
//                 const saveSign = await fetch('http://127.0.0.1:8000/teachers/teacherSign', {
//                   method: 'POST',
//                   body: formData,
//                 })
//                 if (saveSign.ok) {
//                   const response = await saveSign.json();
//                   console.log(response);
//                   if(response?.status_code === 200) {
//                     setShowSignatureModal(false);
//                     setPopupMsg('Signature saved successfully');
//                     setTimeout(() => {
//                       setPopupMsg('');
//                       setShowSignatureModal(false);
//                       window.location.reload();
//                     }, 3000);
//                   }
//                 }
//               }
//               reader.readAsDataURL(file);
//             }
//           };

  
//   // send report to head teacher for conformation
//   const [pdfData, setPdfData] = useState(null);
//   const [generatePdf, setGeneratePdf] = useState(false);


//     const subjectScores = students.map((student) => {
//       const studentSubjectScores = subjects.map((subject) => {
//         const score = studentScoresData.find((score) => score.studentId === student.studentId && score.subjectId === subject.subjectId);
//         if (score === undefined) {
//           return {
//             subjectName: subject.subjectName,
//             score: 0,
//             maxScore: 100,
//             grade: grades.find((grade) => grade.gradeId === 6)?.gradeLetter
//           };
//         }
//         return {
//           subjectName: subject.subjectName,
//           score: score ? score.score : 0,
//           maxScore: 100,
//           grade: grades.find((grade) => grade.gradeId === score?.grade)?.gradeLetter
//         };
//       });
//       return {
//         ...student,
//         subjectScores: studentSubjectScores
//       }
//     });


//     console.log(subjectScores);

//     useEffect(() => {
//       const handleSendReports = async () => {
//         // console.log(pdfData.allPdfs[0]?.blob);

//         // convert blob into file
//         for (let i = 0; i < pdfData.allPdfs.length; i++) {
//           const blob = pdfData.allPdfs[i].blob;
//           const file = new File([blob], 'report.pdf', { type: 'application/pdf' });
//           const formData = new FormData();
//           formData.append('attachment', file);
//           formData.append('studentId', pdfData.allPdfs[i].studentId);
//           formData.append('teacherId', teacher.teacherId);
//           formData.append('comments', comment || 'Student subject wise Report');
//           const response = await fetch('http://127.0.0.1:8000/teachers/uploadReport', {
//             method: 'POST',
//             body: formData
//           });
//           const data = await response.json();
//           console.log(data);
//           if(data?.status_code === 200) {
//             setPopupMsg('Report sent successfully');
//             setTimeout(() => {
//               setPopupMsg('');
//               setShowSendReportModal(false);
//             }, 5000);
//           }else {
//             setPopupMsg('Error sending report');
//             setTimeout(() => setPopupMsg(''), 3000);
//           }
//         }
//       }
//       if(pdfData) {
//         handleSendReports();
//       }
//     }, [pdfData])
  
//     const SendReport = async () => {
//       setGeneratePdf(true);
//     }
    
  
//     const handlePdfGenerated = (pdfInfo) => {
//       console.log("PDF generated successfully:", pdfInfo);
//       setPdfData(pdfInfo);
//       setGeneratePdf(false);
      
//       // You can now:
//       // 1. Automatically download it:
//       // const link = document.createElement('a');
//       // link.href = pdfInfo.url;
//       // link.download = pdfInfo.fileName;
//       // link.click();
      
//       // 2. Send it to a server:
//       // const formData = new FormData();
//       // formData.append('pdf', pdfInfo.blob, pdfInfo.fileName);
//       // fetch('/api/upload-report', { method: 'POST', body: formData });
      
//       // 3. Store it in state for later use
//     };

//     const [openList, setOpenList] = useState(false);
//     const [selectedNotification, setSelectedNotification] = useState(null);

//     console.log(selectedNotification);

//   if (loading) return <Loader />;

//   return (
//     <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-50">
//       {/* Top Navigation Bar */}
//       <div className="bg-white shadow-sm border-b border-gray-200">
//         <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
//           <div className="flex justify-between items-center py-4">
//             <div className="flex items-center space-x-8">
//               <div>
//                 <h1 className="text-2xl font-bold text-gray-900">Class Teacher Dashboard</h1>
//                 <p className="text-sm text-gray-600">
//                   Welcome back, <span className="font-semibold text-indigo-600">{user?.userName || 'Teacher'}</span>
//                 </p>
//               </div>
//             </div>
//             <div className="flex items-center space-x-3">
//               <button
//                 className="relative flex items-center mr-2"
//                 onClick={() => setOpenList(true)}
//               >
//                 <Bell size={18} />

//                 {review.length > 0 && (
//                   <span className="absolute w-2 h-2 bg-red-500 rounded-full top-0 left-3" />
//                 )}
//               </button>

//               {openList && (
//                 <div className="absolute right-1 top-1 mt-2 w-80 bg-white shadow-lg rounded-lg z-50">
//                   <div className="w-full p-3 border-b flex justify-between items-center">
//                     <h2 className="text-lg font-semibold">Notifications</h2>
//                     <button onClick={() => setOpenList(false)} className='font-semibold'>
//                       <X size={18}  />
//                     </button>
//                   </div>

//                   {review.length === 0 ? (
//                     <div className="p-4 text-gray-500 text-sm">
//                       No notifications
//                     </div>
//                   ) : (
//                     review.map((item) => (
//                       <div
//                         onClick={() => setSelectedNotification(item)}
//                         key={item.reviewId}
//                         className="p-3 border-b hover:bg-gray-100 cursor-pointer"
//                       >
//                         <p className="font-medium">
//                           {reports.find(report => report.reportId === item.reportId)?.comments}
//                         </p>
//                         <p className="text-sm text-gray-600 truncate">
//                           {item.comments}
//                         </p>
//                         <span className="text-xs text-gray-400">
//                           {item.created_at}
//                         </span>
//                       </div>
//                     ))
//                   )}
//                 </div>
//               )}


//               {selectedNotification && (
//                 <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50">
//                   <div className="bg-white rounded-lg w-96 p-5">
//                     <h2 className="text-lg font-semibold">
//                       {reports.find(r => r.reportId === selectedNotification.reportId)?.comments}
//                     </h2>

//                     <p className="mt-3 text-gray-700">
//                         Student Name: {students.find(student => reports.find(report => report.reportId === selectedNotification.reportId && report.studentId === student.studentId))?.studentName}
//                       </p>
                    
//                       <p className="mt-3 text-gray-700">
//                         Student Roll No: {students.find(student => reports.find(report => report.reportId === selectedNotification.reportId && report.studentId === student.studentId))?.rollId}
//                       </p>

//                     <p className="mt-3 text-gray-700">
//                       comments: {selectedNotification?.comments}
//                     </p>

//                     <p className="mt-2 text-xs text-gray-400">
//                       {selectedNotification.created_at}
//                     </p>

//                     <button
//                       className="mt-4 px-4 py-2 bg-blue-600 text-white rounded"
//                       onClick={() => setSelectedNotification(null)}
//                     >
//                       Close
//                     </button>
//                   </div>
//                 </div>
//               )}

//               <button
//                 onClick={exportToExcel}
//                 className="flex items-center gap-2 bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded-lg transition-colors font-medium disabled:opacity-50 disabled:cursor-not-allowed"
//                 disabled={filteredStudents.length === 0}
//               >
//                 <Download size={18} />
//                 Export
//               </button>
//               <button
//                 onClick={() => dispatch(logout())}
//                 className="flex items-center gap-2 bg-red-500 hover:bg-red-600 text-white px-4 py-2 rounded-lg transition-colors font-medium"
//               >
//                 <LogOut size={18} />
//                 Logout
//               </button>
//             </div>
//           </div>
//         </div>
//       </div>

//       <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
//         {/* Error and Success Messages */}
//         {errorMessage && (
//           <div className="mb-4 p-4 bg-red-50 border border-red-200 text-red-700 rounded-lg">
//             {errorMessage}
//           </div>
//         )}
        
//         {successMessage && (
//           <div className="mb-4 p-4 bg-green-50 border border-green-200 text-green-700 rounded-lg">
//             {successMessage}
//           </div>
//         )}
        
//         {popUpMsg && (
//           <div className="mb-4 p-4 bg-blue-50 border border-blue-200 text-blue-700 rounded-lg">
//             {popUpMsg}
//           </div>
//         )}

//         {/* Teacher Info Card */}
//         <div className="bg-white rounded-xl shadow-md p-6 mb-6">
//           <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
//             <div className="flex-1">
//               <h2 className="text-lg font-semibold text-gray-800 mb-2">Teaching Profile</h2>
//               <div className="flex flex-wrap gap-2 mb-3">
//                 <span className="inline-flex items-center gap-1 bg-blue-100 text-blue-800 px-3 py-1 rounded-full text-sm">
//                   <GraduationCap size={14} />
//                   Subjects: {teacherSubjects.length}
//                 </span>
//                 <span className="inline-flex items-center gap-1 bg-purple-100 text-purple-800 px-3 py-1 rounded-full text-sm">
//                   <Users size={14} />
//                   Classes: {teacherClasses.length}
//                 </span>
//                 <span className="inline-flex items-center gap-1 bg-green-100 text-green-800 px-3 py-1 rounded-full text-sm">
//                   <Award size={14} />
//                   Total Students: {quickStats.totalStudents}
//                 </span>
//               </div>
//               <div className="flex flex-wrap gap-2">
//                 {teacherSubjects.length > 0 ? (
//                   teacherSubjects.map(subject => (
//                     <span 
//                       key={subject.subjectId} 
//                       className={`px-2 py-1 rounded text-sm cursor-pointer hover:opacity-80 transition ${ 'bg-indigo-50 text-indigo-700'}`}
//                       onClick={() => {
//                         setSignatureSubject(subject);
//                       }}
//                     >
//                       {subject.subjectName}
//                     </span>
//                   ))
//                 ) : (
//                   <span className="text-gray-500 text-sm">No subjects assigned</span>
//                 )}
//               </div>
//             </div>
//           </div>
//         </div>

//         {/* Tab Navigation */}
//         <div className="flex gap-3 mb-6">
//           <button
//             onClick={() => setActiveTab('dashboard')}
//             className={`px-4 py-2 rounded-lg font-medium transition-colors text-white flex items-center gap-2 ${
//               activeTab === 'dashboard' ? 'bg-blue-500' : 'bg-gray-500 hover:bg-gray-600'
//             }`}
//           >
//             <Home size={20} />
//             Dashboard
//           </button>
//           <button
//             onClick={() => setActiveTab('reports')}
//             className={`px-4 py-2 rounded-lg font-medium transition-colors text-white flex items-center gap-2 ${
//               activeTab === 'reports' ? 'bg-blue-500' : 'bg-gray-500 hover:bg-gray-600'
//             }`}
//           >
//             <FileText size={20} />
//             Reports ({reports.length})
//           </button>
//         </div>

//         {/* Dashboard Content */}
//         {activeTab === 'dashboard' && (
//           <>
//             {/* Quick Stats */}
//             <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-6 mb-6">
//               {/* ... (same quick stats cards) ... */}
//             </div>

//             {/* Performance Trends */}
//             {performanceTrend.length > 0 && (
//               <div className="bg-white rounded-xl shadow-sm p-6 mb-6">
//                 <h3 className="text-lg font-semibold text-gray-800 mb-4">Subject Performance Trends</h3>
//                 <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
//                   {performanceTrend.map((trend, index) => (
//                     <div key={index} className="border border-gray-200 rounded-lg p-4">
//                       <div className="flex justify-between items-center mb-2">
//                         <span className="font-medium text-gray-900">{trend.subject}</span>
//                         <span className={`px-2 py-1 rounded-full text-xs font-medium ${
//                           trend.trend === 'up' ? 'bg-green-100 text-green-800' :
//                           trend.trend === 'stable' ? 'bg-yellow-100 text-yellow-800' :
//                           'bg-red-100 text-red-800'
//                         }`}>
//                           {trend.trend === 'up' ? '↑ Improving' : 
//                            trend.trend === 'stable' ? '→ Stable' : '↓ Needs Attention'}
//                         </span>
//                       </div>
//                       <div className="text-2xl font-bold text-gray-900">{trend.averageScore}%</div>
//                       <div className="text-sm text-gray-600">{trend.totalStudents} students</div>
//                     </div>
//                   ))}
//                 </div>
//               </div>
//             )}

//             {/* Students Table Section */}
//           <div className="bg-white rounded-xl shadow-sm overflow-hidden">
//             {/* Filters */}
//             <div className="p-6 border-b border-gray-200">
//               <div className="flex flex-col md:flex-row gap-4 mb-4">
//                 <div className="relative flex-1">
//                   <Search className="absolute left-3 top-3 text-gray-400" size={20} />
//                   <input
//                     type="text"
//                     placeholder="Search students by name..."
//                     value={searchQuery}
//                     onChange={(e) => setSearchQuery(e.target.value)}
//                     className="w-full pl-10 pr-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
//                   />
//                 </div>
//                 <div className="flex gap-2">
//                   <select
//                     value={selectedClass}
//                     onChange={(e) => setSelectedClass(e.target.value)}
//                     className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
//                   >
//                     <option value="All">All Classes</option>
//                     {teacherClasses.map(classItem => (
//                       <option key={classItem.classId} value={classItem.classId}>
//                         {classItem.className}
//                       </option>
//                     ))}
//                   </select>
//                   <select
//                     value={selectedSubject}
//                     onChange={(e) => setSelectedSubject(e.target.value)}
//                     className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
//                   >
//                     <option value="All">All Subjects</option>
//                     {teacherSubjects.map(subject => (
//                       <option key={subject.subjectId} value={subject.subjectId}>
//                         {subject.subjectName}
//                       </option>
//                     ))}
//                   </select>
//                 </div>
//               </div>
//               <div className="flex items-center justify-between text-sm text-gray-600">
//                 <div className="flex items-center justify-between gap-4">
//                   <button
//                     onClick={handleSelectAll}
//                     className="flex items-center gap-2 text-indigo-600 hover:text-indigo-800 font-medium"
//                   >
//                     {selectAll ? <CheckSquare size={16} /> : <Square size={16} />}
//                     {selectAll ? 'Deselect All' : 'Select All Ready'}
//                   </button>
//                   <span>Selected: {selectedStudents.size} | Showing {currentItems.length} of {filteredStudents.length}</span>
//                 </div>
//                 <div>
//                 {selectedStudents.size > 0 && canSendSelectedReports() && (
//                   <button
//                     onClick={() => setShowSendReportModal(true)}
//                     className="flex items-center gap-2 bg-indigo-600 hover:bg-indigo-700 text-white px-4 py-2 rounded-lg transition-colors font-medium"
//                   >
//                     <SendHorizonal size={18} />
//                     Send {selectedStudents.size} Report(s) to Head Teacher
//                   </button>
//                 )}
//                 </div>
//               </div>
//             </div>

//             {/* Table */}
//            <div className="overflow-x-auto">
//              <table className="w-full">
//                <thead className="bg-gray-50">
//                  <tr>
//                    <th className="px-6 py-3 text-left text-xs font-semibold text-gray-600 uppercase">Select</th>
//                    <th className="px-6 py-3 text-left text-xs font-semibold text-gray-600 uppercase">Student</th>
//                    <th className="px-6 py-3 text-left text-xs font-semibold text-gray-600 uppercase">Class</th>
//                    {subjects.map(subject => (
//                     <th key={subject.subjectId} className="px-6 py-3 text-left text-xs font-semibold text-gray-600 uppercase">
//                       <div className="flex items-center justify-between gap-2">
//                         <span>{subject.subjectName}</span>
//                         <button
//                         // disabled = {!mapTeacher.find(teacher => teacher.subjectId === subject.subjectId && teacher.teacherId === teacher.teacherId)}
//                           onClick={() => {
//                             setSignatureSubject(subject);
//                             (Array.from(selectedStudents)).length <= 0 ? alert('Please select at least one student') : setShowSignatureModal(true);
                            
//                           }}
//                           className={`p-1 rounded hover:bg-gray-200 text-gray-400`}
//                           title={signatures[subject.subjectId] ? 'Signed' : 'Not Signed'}
//                         >
//                           <Signature size={16} />
//                         </button>
//                       </div>
//                     </th>
//                   ))}
//                   <th className="px-6 py-3 text-left text-xs font-semibold text-gray-600 uppercase">Status</th>
//                 </tr>
//               </thead>
//               <tbody className="divide-y divide-gray-200">
//                 {currentItems.length > 0 ? (
//                   currentItems.map((student) => {
//                     const isReady = isReadyForConfirmation(student.studentId);
//                     const allFilled = isAllSubjectsFilled(student.studentId);
                    
//                     return (
//                       <tr key={student.studentId} className="hover:bg-gray-50 transition-colors">
//                         <td className="px-6 py-4">
//                           <button
//                             onClick={() => handleSelectStudent(String(student.studentId))}
//                             // disabled={!isReady}
//                             className={`p-1 rounded ${isReady ? 'hover:bg-gray-200' : 'opacity-40 '}`}
//                           >
//                             {selectedStudents.has(String(student.studentId)) ? (
//                               <CheckSquare size={20} className="text-indigo-600" />
//                             ) : (
//                               <Square size={20} className="text-gray-400" />
//                             )}
//                           </button>
//                         </td>
//                         <td className="px-6 py-4">
//                           <div className="flex items-center gap-3">
//                             <div className="w-8 h-8 bg-indigo-100 rounded-full flex items-center justify-center">
//                               <Users className="text-indigo-600" size={16} />
//                             </div>
//                             <div>
//                               <div className="font-medium text-gray-900">{student.studentName}</div>
//                               <div className="text-sm text-gray-500">Roll: {student.rollId}</div>
//                             </div>
//                           </div>
//                         </td>
//                         <td className="px-6 py-4 text-sm text-gray-900">
//                           {getClassDisplayName(student.classId)}
//                         </td>
//                         {subjects.map(subject => {
//                           const score = studentScoresData.find(
//                             s => s.studentId === student.studentId && s.subjectId === subject.subjectId
//                           );
//                           const isSigned = signatures.find(signature => signature.subjectId === subject.subjectId && signature.studentId === student.studentId);
                          
//                           return (
//                             <td key={subject.subjectId} className="px-6 py-4">
//                               <div className="flex flex-col gap-2">
//                                 <div className="flex items-center justify-between gap-2">
//                                   {score ? (
//                                     <div className={`px-3 py-1 rounded text-center text-sm font-medium min-w-[60px] ${
//                                       score.score >= 80 ? 'bg-green-100 text-green-800' :
//                                       score.score >= 60 ? 'bg-yellow-100 text-yellow-800' :
//                                       'bg-red-100 text-red-800'
//                                     }`}>
//                                       {score.score}%
//                                     </div>
//                                   ) : (
//                                     <span className="text-gray-400 text-sm px-3 py-1">N/A</span>
//                                   )}
//                                   <button
//                                     disabled = {!mapTeacher.find(teacher => teacher.subjectId === subject.subjectId && teacher.teacherId === teacher.teacherId)}
//                                     onClick={() => handleAddEditScore(student, subject.subjectId, score)}
//                                     className="px-2 py-1 text-xs bg-indigo-100 text-indigo-700 hover:bg-indigo-200 rounded transition"
//                                   >
//                                     {score ? 'Edit' : 'Add'}
//                                   </button>
//                                 </div>
//                                 {score && isSigned ? (
//                                   <div className="text-xs text-green-600 flex items-center gap-1">
//                                     <Check size={10} />
//                                     Signed
//                                   </div>
//                                 ) : score && !isSigned ? (
//                                   <div className="text-xs text-orange-600 flex items-center gap-1">
//                                     <AlertCircle size={10} />
//                                     Not Signed
//                                   </div>
//                                 ) : null}
//                               </div>
//                             </td>
//                           );
//                         })}
//                         <td className="px-6 py-4">
//                           <div className={`text-xs px-3 py-1 rounded text-center font-medium ${
//                             isReady 
//                               ? 'bg-green-100 text-green-800' 
//                               : allFilled 
//                                 ? 'bg-orange-100 text-orange-800'
//                                 : 'bg-yellow-100 text-yellow-800'
//                           }`}>
//                             {
//                             isAllSubjectsSigned(student.studentId) 
//                               ? 'Signed' : 'Not Signed'}
//                           </div>
//                         </td>
//                       </tr>
//                     );
//                   })
//                 ) : (
//                   <tr>
//                     <td colSpan={4 + teacherSubjects.length} className="px-6 py-8 text-center text-gray-500">
//                       No students match your filters
//                     </td>
//                   </tr>
//                 )}
//               </tbody>
//             </table>
//           </div>
//           </div>


//           {/* Pagination */}
//           {currentItems.length > 0 && (
//             <div className="px-6 py-4 border-t border-gray-200 bg-gray-50">
//               <div className="flex flex-col sm:flex-row justify-between items-center gap-4">
//                 <div className="flex items-center gap-4">
//                   <div className="flex items-center gap-2">
//                     <span className="text-sm text-gray-600">Show:</span>
//                     <select
//                       value={itemsPerPage}
//                       onChange={(e) => {
//                         setItemsPerPage(Number(e.target.value));
//                         setCurrentPage(1);
//                       }}
//                       className="text-sm border border-gray-300 rounded px-2 py-1 bg-white"
//                     >
//                       <option value="5">5</option>
//                       <option value="10">10</option>
//                       <option value="25">25</option>
//                       <option value="50">50</option>
//                     </select>
//                     <span className="text-sm text-gray-600">per page</span>
//                   </div>
//                   <div className="text-sm text-gray-600">
//                     Page {currentPage} of {totalPages}
//                   </div>
//                 </div>
                
//                 <div className="flex items-center gap-2">
//                   <button
//                     onClick={() => handlePageChange(currentPage - 1)}
//                     disabled={currentPage === 1}
//                     className={`px-3 py-1 rounded border ${currentPage === 1 ? 'text-gray-400 border-gray-300 cursor-not-allowed' : 'text-gray-700 border-gray-300 hover:bg-gray-100'}`}
//                   >
//                     <ChevronLeft size={16} />
//                   </button>
                  
//                   {getPageNumbers().map((page, index) => (
//                     <button
//                       key={index}
//                       onClick={() => typeof page === 'number' ? handlePageChange(page) : null}
//                       className={`px-3 py-1 rounded border ${
//                         currentPage === page 
//                           ? 'bg-indigo-600 text-white border-indigo-600' 
//                           : 'text-gray-700 border-gray-300 hover:bg-gray-100'
//                       } ${typeof page !== 'number' ? 'cursor-default hover:bg-transparent' : ''}`}
//                       disabled={typeof page !== 'number'}
//                     >
//                       {page}
//                     </button>
//                   ))}
                  
//                   <button
//                     onClick={() => handlePageChange(currentPage + 1)}
//                     disabled={currentPage === totalPages}
//                     className={`px-3 py-1 rounded border ${currentPage === totalPages ? 'text-gray-400 border-gray-300 cursor-not-allowed' : 'text-gray-700 border-gray-300 hover:bg-gray-100'}`}
//                   >
//                     <ChevronRight size={16} />
//                   </button>
//                 </div>
//               </div>
//             </div>
//           )}
//           </>
//         )}

//         {/* Reports Tab */}
//         {activeTab === 'reports' && (
//           <div className="bg-white rounded-xl shadow-sm overflow-hidden">
//             <div className="p-6 border-b border-gray-200">
//               <div className="flex justify-between items-center">
//                 <h3 className="text-lg font-semibold text-gray-800">Submitted Reports</h3>
//               </div>
//             </div>
//             <div className="p-6">
//               {reports.length > 0 ? (
//                 reports.map(report => (
//                   <div
//                     key={report.reviewId || report.id}
//                     onClick={() => handleReportClick(report)}
//                     className="flex items-center bg-white mb-3 justify-between p-4 border border-gray-200 rounded-lg hover:bg-gray-50 cursor-pointer transition"
//                   >
//                     <div className="flex-1">
//                       <h3 className="font-semibold text-gray-800">
//                         {report.comments || 'Report'}
//                       </h3>
//                       <p className="text-sm text-gray-600">
//                         Teacher: {report.teacherId === teacher.teacherId ? teacher.teacherName : 'Unknown'}
//                       </p>
//                       <p className="text-sm text-gray-500">
//                         Submitted: {new Date(report.createdAt || Date.now()).toLocaleDateString()}
//                       </p>
//                       <p className={`text-sm text-gray-500 flex items-center`}>
//                         Status: <p className={`font-bold ${report.status ? 'text-green-600' : 'text-red-600'}`}>{report.status ? 'Approved' : 'Pending'}</p>
//                       </p>
//                     </div>
//                     <div className="flex items-center gap-3">
//                       {/* <span className={`px-3 py-1 rounded-full text-sm font-medium ${getStatusColor(report.iStatus)}`}>
//                         {getStatusText(report.iStatus)}
//                       </span> */}
//                       {report.comments && (
//                         <MessageSquare size={20} className="text-blue-600" />
//                       )}
//                     </div>
//                   </div>
//                 ))
//               ) : (
//                 <div className="text-center py-8 text-gray-500">
//                   <FileText size={48} className="mx-auto mb-4 text-gray-300" />
//                   <p className="text-lg font-medium mb-2">No reports submitted yet</p>
//                   <p className="text-gray-600">Upload your first report to get started</p>
//                 </div>
//               )}
//             </div>
//           </div>
//         )}

//         {/* Send Report Modal */}
//         {showSendReportModal && (
//           <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
//             <div className="bg-white rounded-xl shadow-2xl max-w-md w-full">
//               <div className="p-6 border-b border-gray-200">
//                 <div className="flex justify-between items-center">
//                   <h3 className="text-lg font-semibold text-gray-900">Send Reports to Head Teacher</h3>
//                   <button
//                     onClick={() => setShowSendReportModal(false)}
//                     className="text-gray-400 hover:text-gray-600 transition p-1 rounded-full hover:bg-gray-100"
//                   >
//                     <X size={24} />
//                   </button>
//                 </div>
//               </div>
//               <div className="p-6">
//                 <div className="mb-6">
//                 {popUpMsg && <p className="text-sm text-green-600 mt-1 mb-1">{popUpMsg}</p>}
//                   <div className="flex items-center gap-2 mb-4">
//                     <SendHorizonal className="text-indigo-600" size={24} />
//                     <p className="font-medium text-gray-900">
//                       Sending {selectedStudents.size} report(s) to head teacher
//                     </p>
//                   </div>
                  
//                   {
//                     comments.length > 0 && (
//                       <div className="bg-blue-50 p-4 rounded-lg mb-4">
//                         <p className="text-sm text-blue-800 mb-2">
//                           <strong>Note:</strong> The following comments will be included:
//                         </p>
//                         <ul className="text-sm text-blue-800 space-y-1">
//                           {comments.map((comment, index) => (
//                             <li key={index} className="flex items-center gap-2">
//                               <Check className="text-green-600" size={16} />
//                               {comment}
//                             </li>
//                           ))}
//                         </ul>
//                       </div>
//                     )
//                   }
                  
//                   {/* <div className="bg-blue-50 p-4 rounded-lg mb-4">
//                     <p className="text-sm text-blue-800 mb-2">
//                       <strong>Note:</strong> The following signatures will be included:
//                     </p>
//                     <ul className="text-sm text-blue-800 space-y-1">
//                       {subjects.map(subject => (
//                         <li key={subject.subjectId} className="flex items-center gap-2">
//                           {signatures[subject.subjectId] ? (
//                             <Check className="text-green-600" size={16} />
//                           ) : (
//                             <X className="text-red-600" size={16} />
//                           )}
//                           {subject.subjectName}: {signatures[subject.subjectId] ? 'Signed' : 'Not Signed'}
//                         </li>
//                       ))}
//                     </ul>
//                   </div> */}
//                 </div>
//                 <div className="flex gap-3">
//                   <button
//                     onClick={() => setShowSendReportModal(false)}
//                     className="flex-1 px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition"
//                     disabled={sendingReports}
//                   >
//                     Cancel
//                   </button>
//                   <button
//                     onClick={SendReport}
//                     disabled={sendingReports}
//                     className={`flex-1 px-4 py-2 rounded-lg transition flex items-center justify-center gap-2 ${
//                       sendingReports
//                         ? 'bg-gray-400 cursor-not-allowed'
//                         : 'bg-indigo-600 hover:bg-indigo-700 text-white'
//                     }`}
//                   >
//                     {sendingReports ? (
//                       <>
//                         <Loader size="sm" />
//                         Sending...
//                       </>
//                     ) : (
//                       <>
//                         <SendHorizonal size={18} />
//                         Send Reports
//                       </>
//                     )}
//                   </button>
//                 </div>
//               </div>
//             </div>
//           </div>
//         )}

//         {/* Score Modal (Updated) */}
//         {showScoreModal && selectedStudent && (
//           <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
//             <div className="bg-white rounded-xl shadow-2xl max-w-md w-full">
//               <div className="p-6 border-b border-gray-200">
//                 <div className="flex justify-between items-center">
//                   <div>
//                     <h3 className="text-lg font-semibold text-gray-900">
//                       {selectedScoreForEdit ? 'Edit Score' : 'Add Score'}
//                     </h3>
//                     <p className="text-sm text-gray-600 mt-1">
//                       {selectedStudent.studentName} • {
//                         teacherSubjects.find(s => s.subjectId === Number(scoreData.subject))?.subjectName || 'Select Subject'
//                       }
//                     </p>
//                   </div>
//                   <button
//                     onClick={() => {
//                       setShowScoreModal(false);
//                       setSelectedStudent(null);
//                       setScoreData({ subject: '', score: '', student_id: '' });
//                       setSelectedScoreForEdit(null);
//                     }}
//                     className="text-gray-400 hover:text-gray-600 transition p-1 rounded-full hover:bg-gray-100"
//                   >
//                     <X size={24} />
//                   </button>
//                 </div>
//               </div>
//               <div className="p-6">
//                 {successMessage && (
//                   <div className="mb-4 p-3 bg-green-50 text-green-700 rounded-lg">
//                     {successMessage}
//                   </div>
//                 )}
//                 {errorMessage && (
//                   <div className="mb-4 p-3 bg-red-50 text-red-700 rounded-lg">
//                     {errorMessage}
//                   </div>
//                 )}
//                 <div className="space-y-4">
//                   {!selectedScoreForEdit && (
//                     <div>
//                       <label className="block text-sm font-medium text-gray-700 mb-2">Subject</label>
//                       <select
//                         value={scoreData.subject}
//                         onChange={(e) => setScoreData({ ...scoreData, subject: e.target.value })}
//                         className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
//                         required
//                       >
//                         <option value="">Select Subject</option>
//                         {teacherSubjects.map((subject) => (
//                           <option key={subject.subjectId} value={subject.subjectId}>
//                             {subject.subjectName}
//                           </option>
//                         ))}
//                       </select>
//                     </div>
//                   )}
//                   <div>
//                     <label className="block text-sm font-medium text-gray-700 mb-2">Score (0-100)</label>
//                     <input
//                       type="number"
//                       min="0"
//                       max="100"
//                       maxLength="3"
//                       value={scoreData.score}
//                       onChange={(e) => {
//                         const inputValue = e.target.value;
//                         const parsedValue = parseInt(inputValue, 10);
                    
//                         if (!isNaN(parsedValue) && parsedValue >= 0 && parsedValue <= 100 && inputValue.length <= 3) {
//                           setScoreData({ ...scoreData, score: parsedValue });
//                         }
//                       }}
//                       className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
//                       placeholder="Enter score"
//                       required
//                     />
//                   </div>
//                 </div>
//                 <div className="flex gap-3 mt-6">
//                   <button
//                     onClick={() => {
//                       setShowScoreModal(false);
//                       setSelectedStudent(null);
//                       setScoreData({ subject: '', score: '', student_id: '' });
//                       setSelectedScoreForEdit(null);
//                     }}
//                     className="flex-1 px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition"
//                   >
//                     Cancel
//                   </button>
//                   <button
//                     onClick={handleSaveScore}
//                     className="flex-1 px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition disabled:opacity-50 disabled:cursor-not-allowed"
//                     disabled={!scoreData.subject || !scoreData.score}
//                   >
//                     {selectedScoreForEdit ? 'Update Score' : 'Save Score'}
//                   </button>
//                 </div>
//               </div>
//             </div>
//           </div>
//         )}

//         {/* Report Viewer Modal */}
//         {selectedReport && (
//           <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
//              <div className="bg-white rounded-xl shadow-2xl max-w-4xl w-full max-h-[90vh] overflow-hidden flex flex-col">
//                <div className="bg-indigo-600 text-white p-4 flex items-center justify-between">
//                  <h3 className="text-xl font-bold">Report Details</h3>
//                  <button
//                   onClick={() => setSelectedReport(null)}
//                   className="text-white hover:text-gray-200 p-1 rounded-full hover:bg-indigo-700"
//                 >
//                   <X size={24} />
//                 </button>
//               </div>
              
//               <div className="p-6 overflow-y-auto flex-1">
//                 <div className="mb-4 flex items-center gap-4">
//                   <button
//                     onClick={() => setZoomLevel(Math.max(50, zoomLevel - 10))}
//                     className="flex items-center gap-2 px-3 py-2 bg-gray-200 rounded-lg hover:bg-gray-300"
//                   >
//                     <ZoomOut size={18} />
//                     Zoom Out
//                   </button>
//                   <span className="font-medium">{zoomLevel}%</span>
//                   <button
//                     onClick={() => setZoomLevel(Math.min(200, zoomLevel + 10))}
//                     className="flex items-center gap-2 px-3 py-2 bg-gray-200 rounded-lg hover:bg-gray-300"
//                   >
//                     <ZoomIn size={18} />
//                     Zoom In
//                   </button>
//                 </div>
          
//                 {selectedReport.report ? (
//                   <div
//                     className="border-2 border-gray-300 rounded-lg p-8 bg-gray-50 mb-6 transition-transform overflow-auto"
//                     style={{ transform: `scale(${zoomLevel / 100})`, transformOrigin: 'top left' }}
//                   >
//                     <MyPdfViewer pdfBlobUrl={selectedReport.report} />
//                   </div>
//                 ) : (
//                   <div className="border-2 border-gray-300 rounded-lg p-8 bg-gray-50 mb-6 text-center">
//                     <FileText size={48} className="mx-auto mb-4 text-gray-400" />
//                     <p className="text-gray-600">No document preview available</p>
//                   </div>
//                 )}
          
//                 {selectedReport.iStatus === undefined || selectedReport.iStatus === null ? (
//                   <div className="space-y-4">
//                     <div>
//                       <label className="block text-sm font-medium text-gray-700 mb-2">
//                         Comments (Optional for Approval, Required for Rejection)
//                       </label>
//                       <textarea
//                         value={comment}
//                         onChange={(e) => setComment(e.target.value)}
//                         placeholder="Add your comments here..."
//                         className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
//                         rows="4"
//                       />
//                     </div>
          
//                     <div className="flex gap-4">
//                       <button
//                         onClick={handleApprove}
//                         className="flex-1 flex items-center justify-center gap-2 px-6 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 transition"
//                       >
//                         <CheckCircle size={20} />
//                         Approve Report
//                       </button>
//                       <button
//                         onClick={handleReject}
//                         className="flex-1 flex items-center justify-center gap-2 px-6 py-3 bg-red-600 text-white rounded-lg hover:bg-red-700 transition"
//                       >
//                         <XCircle size={20} />
//                         Reject Report
//                       </button>
//                     </div>
//                   </div>
//                 ) : (
//                   <div className={`p-4 rounded-lg ${
//                     selectedReport.iStatus ? 'bg-green-50 border border-green-200' : 'bg-red-50 border border-red-200'
//                   }`}>
//                     <p className="font-semibold mb-2">
//                       Status: {getStatusText(selectedReport.iStatus)}
//                     </p>
//                     {selectedReport.comments && (
//                       <div>
//                         <p className="text-sm font-medium text-gray-700 mb-1">Comments:</p>
//                         <p className="text-gray-700 italic">{selectedReport.comments}</p>
//                       </div>
//                     )}
//                   </div>
//                 )}
//               </div>
//             </div>
//           </div>
//         )}

//         {/* Signature Modal */}
//       {showSignatureModal && (
//         <Modal
//           title="Add Signature"
//           onClose={() => setShowSignatureModal(false)
//           }
//         >
//           <div className="space-y-6">
//             <div>
//               <h3 className="font-semibold text-gray-900 mb-3">Draw Signature</h3>
//               <div className="border-2 border-gray-300 rounded-xl overflow-hidden bg-gray-50">
//                 <canvas
//                   ref={canvasRef}
//                   width={450}
//                   height={200}
//                   className="w-full cursor-crosshair"
//                   onMouseDown={startDrawing}
//                   onMouseMove={draw}
//                   onMouseUp={stopDrawing}
//                   onMouseLeave={stopDrawing}
//                 />
//               </div>
//               <button
//                 onClick={clearSignature}
//                 className="mt-2 text-sm text-red-600 hover:text-red-700 font-medium"
//               >
//                 Clear Canvas
//               </button>
//             </div>

//             <div>
//               <h3 className="font-semibold text-gray-900 mb-3">Or Upload Image</h3>
//               <label className="flex items-center justify-center gap-2 border-2 border-dashed border-gray-300 rounded-xl p-6 cursor-pointer hover:border-blue-500 transition bg-gray-50">
//                 <Upload size={20} className="text-gray-400" />
//                 <span className="text-gray-600">Upload signature image</span>
//                 <input
//                   type="file"
//                   accept="image/*"
//                   onChange={handleSignatureUpload}
//                   className="hidden"
//                 />
//               </label>
//             </div>

//             <div className="flex gap-3 justify-end">
//               <button
//                 onClick={() => setShowSignatureModal(false)}
//                 className="px-6 py-2 border border-gray-300 rounded-lg hover:bg-gray-100 transition"
//               >
//                 Cancel
//               </button>
//               <button
//                 onClick={saveSignature}
//                 className="flex items-center gap-2 bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 transition"
//               >
//                 <Check size={20} />
//                 Save Signature
//               </button>
//             </div>
//           </div>
//         </Modal>
//       )}

// {generatePdf && (
//   <BatchStudentReportCard
//     schoolDetail={school}
//     studentsData={subjectScores.filter(student => (Array.from(selectedStudents)).includes(student.studentId.toString()))}
//     onBatchComplete={handlePdfGenerated}
//   />
// )}


//         {/* Upload Report Modal */}
//         {showUploadModal && (
//           <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
//              <div className="bg-white rounded-xl shadow-2xl max-w-md w-full">
//                <div className="p-6 border-b border-gray-200">
//                  <div className="flex justify-between items-center">
//                    <h3 className="text-lg font-semibold text-gray-900">Upload Report</h3>
//                    <button
//                     onClick={() => {
//                       setShowUploadModal(false);
//                       setUploadedReport(null);
//                       setComment('');
//                     }}
//                     className="text-gray-400 hover:text-gray-600 transition p-1 rounded-full hover:bg-gray-100"
//                   >
//                     <X size={24} />
//                   </button>
//                 </div>
//               </div>
//               <div className="p-6">
//                 <div className="mb-6">
//                   <label className="block text-sm font-medium text-gray-700 mb-2">Report Comments</label>
//                   <textarea
//                     value={comment}
//                     onChange={(e) => setComment(e.target.value)}
//                     placeholder="Enter report description or comments..."
//                     className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent mb-4"
//                     rows="3"
//                   />
                  
//                   <label className="block text-sm font-medium text-gray-700 mb-2">Select Report File</label>
//                   <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-indigo-500 transition-colors cursor-pointer">
//                     <input
//                       type="file"
//                       accept=".pdf,.doc,.docx"
//                       onChange={(e) => {
//                         const file = e.target.files[0];
//                         if (file && file.size > 10 * 1024 * 1024) {
//                           setErrorMessage('File size must be less than 10MB');
//                           return;
//                         }
//                         setUploadedReport(file);
//                         setErrorMessage('');
//                       }}
//                       className="hidden"
//                       id="report-upload"
//                     />
//                     <label htmlFor="report-upload" className="cursor-pointer">
//                       <Upload className="mx-auto text-gray-400 mb-3" size={40} />
//                       <p className="text-gray-600 mb-2">Click to upload or drag and drop</p>
//                       <p className="text-sm text-gray-500">PDF, DOC, DOCX (max 10MB)</p>
//                     </label>
//                   </div>
//                   {uploadedReport && (
//                     <div className="mt-3 p-3 bg-green-50 rounded-lg flex justify-between items-center">
//                       <p className="text-green-700 truncate">{uploadedReport.name}</p>
//                       <button
//                         onClick={() => setUploadedReport(null)}
//                         className="text-red-500 hover:text-red-700"
//                       >
//                         <X size={18} />
//                       </button>
//                     </div>
//                   )}
//                 </div>
//                 <div className="flex gap-3">
//                   <button
//                     onClick={() => {
//                       setShowUploadModal(false);
//                       setUploadedReport(null);
//                       setComment('');
//                     }}
//                     className="flex-1 px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition"
//                   >
//                     Cancel
//                   </button>
//                   <button
//                     onClick={handleUploadReport}
//                     disabled={!uploadedReport}
//                     className={`flex-1 px-4 py-2 rounded-lg transition ${
//                       uploadedReport
//                         ? 'bg-indigo-600 text-white hover:bg-indigo-700'
//                         : 'bg-gray-200 text-gray-500 cursor-not-allowed'
//                     }`}
//                   >
//                     Upload Report
//                   </button>
//                 </div>
//               </div>
//             </div>
//           </div>
//         )}
//       </div>
//     </div>
//   );
// };

// const Modal = ({ title, children, onClose, width = 'max-w-lg' }) => (
//   <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
//     <div className={`bg-white rounded-2xl shadow-2xl ${width} w-full max-h-[90vh] overflow-hidden`}>
//       <div className="bg-gradient-to-r from-indigo-600 to-purple-600 text-white p-6 flex items-center justify-between">
//         <h2 className="text-2xl font-bold">{title}</h2>
//         <button
//           onClick={onClose}
//           className="text-white hover:text-gray-200 text-2xl"
//         >
//           ✕
//         </button>
//       </div>
//       <div className="p-6 overflow-y-auto max-h-[calc(90vh-80px)]">
//         {children}
//       </div>
//     </div>
//   </div>
// );

// export default ClassTeacherDashboard;

import React, { useState, useMemo, useEffect, useCallback, useRef } from 'react';
import { 
  Search, Download, X, Plus, Send, FileText, ZoomIn, ZoomOut, 
  CheckCircle, XCircle, MessageSquare, Check, Upload, LogOut,
  BarChart3, TrendingUp, Users, GraduationCap, BookOpen, Clock,
  Calendar, Award, Star, AlertCircle, Bell, Filter, ChevronRight,
  ChevronDown, PieChart, Target, Bookmark, UserCheck, ShieldCheck,
  MessageCircle, ThumbsUp, Eye, Edit2, Trash2, MoreVertical,
  Home, CheckSquare, Square, PenTool, Signature, SendHorizonal,
  ChevronLeft
} from 'lucide-react';
import { useDispatch, useSelector } from 'react-redux';
import { useNavigate } from 'react-router-dom';
import ExcelJS from 'exceljs';
import Loader from './loader';
import MyPdfViewer from './pdfViewer';
import SubjectReportCard from './subjectReport';
import { initializeAuth, logout } from '../authSlice';
import BatchStudentReportCard from './subjectReport';

const ClassTeacherDashboard = () => {
  const { user } = useSelector((state) => state.auth);
  console.log('User:', user);
  const navigate = useNavigate();
  const dispatch = useDispatch();

  useEffect(() => {
    dispatch(initializeAuth());
  }, [dispatch]);

  // State management
  const [subjects, setSubjects] = useState([]);
  const [classes, setClasses] = useState([]);
  const [students, setStudents] = useState([]);
  const [teacherSubjects, setTeacherSubjects] = useState([]);
  const [studentScoresData, setStudentScoresData] = useState([]);
  const [teacherClasses, setTeacherClasses] = useState([]);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('dashboard');
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedStatus, setSelectedStatus] = useState('All');
  const [selectedClass, setSelectedClass] = useState('All');
  const [selectedSubject, setSelectedSubject] = useState('All');
  const [showScoreModal, setShowScoreModal] = useState(false);
  const [selectedStudent, setSelectedStudent] = useState(null);
  const [selectedScoreForEdit, setSelectedScoreForEdit] = useState(null);
  const [scoreData, setScoreData] = useState({ subject: '', score: '', student_id: '' });
  const [successMessage, setSuccessMessage] = useState('');
  const [errorMessage, setErrorMessage] = useState('');
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [uploadedReport, setUploadedReport] = useState(null);
  const [popUpMsg, setPopupMsg] = useState('');
  const [reports, setReports] = useState([]);
  const [teacher, setTeacher] = useState({});
  const [mapTeacher, setMapTeacher] = useState([]);
  const [quickStats, setQuickStats] = useState({
    totalStudents: 0,
    averageScore: 0,
    topPerformers: 0,
    pendingScores: 0,
    recentSubmissions: 0
  });
  const [performanceTrend, setPerformanceTrend] = useState([]);
  const [grades, setGrades] = useState([]);
  const [school, setSchool] = useState({});
  
  // New states for inline editing
  const [editingCell, setEditingCell] = useState(null); // { studentId, subjectId }
  const [tempScore, setTempScore] = useState('');
  const [savingCells, setSavingCells] = useState(new Set());
  
  // Notification states
  const [seenNotifications, setSeenNotifications] = useState(new Set());
  const [notificationCount, setNotificationCount] = useState(0);
  
  // Signature states
  const [signatureSubject, setSignatureSubject] = useState([]);
  const [uploadedSignature, setUploadedSignature] = useState(null);
  const [signatures, setSignatures] = useState({});
  const [selectedStudents, setSelectedStudents] = useState(new Set());
  const [selectAll, setSelectAll] = useState(false);
  const [showSendReportModal, setShowSendReportModal] = useState(false);
  const [sendingReports, setSendingReports] = useState(false);
  const [comments, setComments] = useState([]);
  const [review, setReviews] = useState([]);
  
  // Other states
  const [selectedReport, setSelectedReport] = useState(null);
  const [zoomLevel, setZoomLevel] = useState(100);
  const [comment, setComment] = useState('');
  const [showSuccess, setShowSuccess] = useState(false);
  const [signature, setSignature] = useState(null);
  const [showSignatureModal, setShowSignatureModal] = useState(false);
  const [isDrawing, setIsDrawing] = useState(false);
  const canvasRef = useRef(null);
  const [pdfData, setPdfData] = useState(null);
  const [generatePdf, setGeneratePdf] = useState(false);
  const [openList, setOpenList] = useState(false);
  const [selectedNotification, setSelectedNotification] = useState(null);

  // Pagination
  const [currentPage, setCurrentPage] = useState(1);
  const [itemsPerPage, setItemsPerPage] = useState(5);

  // Fetch teacher data
  useEffect(() => {
    const fetchTeacher = async () => {
      try {
        if (!user?.userEmail) return;
        
        const res = await fetch(
          `http://127.0.0.1:8000/teachers/email?email=${user.userEmail}`
        );
        
        if (!res.ok) {
          throw new Error(`HTTP error! status: ${res.status}`);
        }
        
        const data = await res.json();
        setTeacher(data?.data);
      } catch (err) {
        console.error("Error fetching teacher:", err);
      }
    };
  
    fetchTeacher();
  }, [user]);

  // Load seen notifications from localStorage
  useEffect(() => {
    const seen = JSON.parse(localStorage.getItem('seenNotifications') || '[]');
    setSeenNotifications(new Set(seen));
  }, []);

  // Fetch all data after teacher is loaded
  useEffect(() => {
    const fetchData = async () => {
      if (!teacher?.teacherId) return;
      
      setLoading(true);
      try {
        const teacherId = Number(teacher.teacherId);
        
        // Fetch all data in parallel
        const [
          studentResponse, 
          subjectResponse, 
          classResponse, 
          studentScoreResponse,
          workFlowResponse,
          MapResponse,
          teacherSign,
          gradesresponse,
          schoolResponse
        ] = await Promise.all([
          fetch('http://127.0.0.1:8000/students/'),
          fetch('http://127.0.0.1:8000/subjects/'),
          fetch('http://127.0.0.1:8000/classes/'),
          fetch(`http://127.0.0.1:8000/students/score`),
          fetch(`http://127.0.0.1:8000/teachers/getWorkFlow?teacherId=${teacherId}`),
          fetch(`http://127.0.0.1:8000/teachers/class_and_subjects?teacherId=${teacherId}`),
          fetch(`http://127.0.0.1:8000/teachers/teacherSign/`),
          fetch(`http://127.0.0.1:8000/grades/`),
          fetch(`http://127.0.0.1:8000/admin/schools/${teacher?.schoolId}`)
        ]);

        // Check all responses
        if (!studentResponse.ok) throw new Error('Failed to fetch students');
        if (!subjectResponse.ok) throw new Error('Failed to fetch subjects');
        if (!classResponse.ok) throw new Error('Failed to fetch classes');
        if (!studentScoreResponse.ok) throw new Error('Failed to fetch scores');
        if (!workFlowResponse.ok) throw new Error('Failed to fetch workflow');
        if (!MapResponse.ok) throw new Error('Failed to fetch teacher mappings');
        if (!teacherSign.ok) throw new Error('Failed to fetch teacher signature');
        if (!gradesresponse.ok) throw new Error('Failed to fetch grades');
        if (!schoolResponse.ok) throw new Error('Failed to fetch school');

        const [
          studentData, 
          subjectData, 
          classData, 
          studentScores,
          workflowData,
          mapData,
          teacherSignData,
          grade,
          school
        ] = await Promise.all([
          studentResponse.json(),
          subjectResponse.json(),
          classResponse.json(),
          studentScoreResponse.json(),
          workFlowResponse.json(),
          MapResponse.json(),
          teacherSign.json(),
          gradesresponse.json(),
          schoolResponse.json()
        ]);

        setSchool(school?.data || {});
        setStudentScoresData(studentScores?.data || []);
        setGrades(grade?.data || []);
        setClasses(classData?.data || []);
        setSignatures(teacherSignData?.data || {});
        setMapTeacher(mapData?.data || []);

        // Filter students by school
        const filteredStudents = (studentData?.data || []).filter(student => 
          student.schoolId === teacher.schoolId
        );

        
      
        setStudents(filteredStudents || []);
        setSubjects(subjectData?.data);

        // Fetch reviews
        try {
          const params = new URLSearchParams();
          (workflowData?.data || []).forEach(r => {
            params.append("reportIds", r.reportId);
          });
      
          const response = await fetch(
            `http://127.0.0.1:8000/teachers/getWorkFlow/ids?${params.toString()}`
          );
      
          const data = await response.json();
          setReviews(data?.data || []);
        } catch (error) {
          console.error("Error fetching reviews:", error);
        }

        // Extract unique subjects and classes from map data
        const uniqueSubjectIds = [...new Set((mapData?.data || []).map(item => item.subjectId))];
        const uniqueClassIds = [...new Set((mapData?.data || []).map(item => item.classId))];

        const teacherSubjectsData = (subjectData?.data || []).filter(subject => 
          uniqueSubjectIds.includes(subject.subjectId)
        );
        
        const teacherClassesData = (classData?.data || []).filter(cls => 
          uniqueClassIds.includes(cls.classId)
        );

        setTeacherSubjects(teacherSubjectsData);
        setTeacherClasses(teacherClassesData);

        // Process workflow attachments
        const processedWorkflow = (workflowData?.data || []).map(item => {
          if (item.report) {
            try {
              const base64 = item.report.replace(/\s/g, "");
              const byteCharacters = atob(base64);
              const byteArray = new Uint8Array(byteCharacters.length);
              for (let i = 0; i < byteCharacters.length; i++) {
                byteArray[i] = byteCharacters.charCodeAt(i);
              }
              const blob = new Blob([byteArray], { type: "application/pdf" });
              return {
                ...item,
                report: URL.createObjectURL(blob)
              };
            } catch (error) {
              console.error("Error processing report:", error);
              return item;
            }
          }
          return item;
        });
        
        setReports(processedWorkflow);

        // Calculate quick stats
        const totalStudents = filteredStudents.length;
        const scores = studentScores?.data || [];
        const avgScore = totalStudents > 0 && scores.length > 0
          ? scores.reduce((sum, score) => sum + (score.score || 0), 0) / scores.length
          : 0;

        const topPerformers = filteredStudents.filter(student => {
          const studentScores = scores.filter(s => s.studentId === student.studentId);
          if (studentScores.length === 0) return false;
          const avg = studentScores.reduce((sum, s) => sum + (s.score || 0), 0) / studentScores.length;
          return avg >= 80;
        }).length;

        const pendingScores = Math.max(
          0, 
          filteredStudents.length * teacherSubjectsData.length - scores.length
        );

        setQuickStats({
          totalStudents,
          averageScore: Math.round(avgScore * 100) / 100,
          topPerformers,
          pendingScores,
          recentSubmissions: processedWorkflow.length
        });

        // Calculate performance trends
        const trends = subjects.map(subject => {
          const subjectScores = scores.filter(s => s.subjectId === subject.subjectId);
          const avg = subjectScores.length > 0
            ? subjectScores.reduce((sum, s) => sum + (s.score || 0), 0) / subjectScores.length
            : 0;
          
          return {
            subject: subject.subjectName,
            averageScore: Math.round(avg),
            totalStudents: filteredStudents.filter(s => 
              (mapData?.data || []).some(m => m.classId === s.classId && m.subjectId === subject.subjectId)
            ).length,
            trend: avg >= 75 ? 'up' : avg >= 60 ? 'stable' : 'down'
          };
        });

        setPerformanceTrend(trends);

      } catch (error) {
        console.error('Error fetching data:', error);
        setErrorMessage('Failed to load data. Please try again.');
        setTimeout(() => {
          setErrorMessage('');
        }, 3000);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [teacher]);

  // Filter unread notifications
  const unreadNotifications = useMemo(() => {
    return review.filter(notification => !seenNotifications.has(notification.reviewId));
  }, [review, seenNotifications]);

  // Update notification count
  useEffect(() => {
    setNotificationCount(unreadNotifications.length);
  }, [unreadNotifications]);

  // Filter students
  const filteredStudents = useMemo(() => {
    if (!students.length) return [];
    
    return students.filter(student => {
      const matchesSearch = searchQuery === '' || 
        student.studentName?.toLowerCase().includes(searchQuery.toLowerCase());
      
      const matchesClass = selectedClass === 'All' || 
        String(student.classId) === selectedClass;
      
      const studentInTeacherClass = teacherClasses.some(cls => 
        cls.classId === student.classId
      );
      
      const matchesSubject = selectedSubject === 'All' || 
        teacherSubjects.some(subject => 
          subject.subjectId === Number(selectedSubject)
        );
      
      const matchesStatus = selectedStatus === 'All' || 
        String(student.active) === selectedStatus;
      
      return matchesSearch && matchesClass && matchesSubject && matchesStatus && studentInTeacherClass;
    });
  }, [students, searchQuery, selectedClass, selectedSubject, selectedStatus, teacherSubjects, teacherClasses]);

  // Pagination calculations
  const indexOfLastItem = currentPage * itemsPerPage;
  const indexOfFirstItem = indexOfLastItem - itemsPerPage;
  const currentItems = filteredStudents.slice(indexOfFirstItem, indexOfLastItem);
  const totalPages = Math.ceil(filteredStudents.length / itemsPerPage);

  // Reset to first page when filters change
  useEffect(() => {
    setCurrentPage(1);
  }, [searchQuery, selectedClass, selectedSubject]);

  // Handle page change
  const handlePageChange = (pageNumber) => {
    if (pageNumber >= 1 && pageNumber <= totalPages) {
      setCurrentPage(pageNumber);
    }
  };

  const getPageNumbers = () => {
    const pages = [];
    const maxPagesToShow = 5;
  
    if (totalPages <= maxPagesToShow) {
      for (let i = 1; i <= totalPages; i++) pages.push(i);
    } else {
      pages.push(1);
      if (currentPage > 3) pages.push('...');
      const start = Math.max(2, currentPage - 1);
      const end = Math.min(totalPages - 1, currentPage + 1);
      for (let i = start; i <= end; i++) pages.push(i);
      if (currentPage < totalPages - 2) pages.push('...');
      pages.push(totalPages);
    }
    return pages;
  };

  // INLINE EDITING FUNCTIONS

  // Handle inline score update
  const handleInlineScoreUpdate = (studentId, subjectId, currentScore = '') => {
    const canEdit = mapTeacher.find(t => 
      t.subjectId === subjectId && t.teacherId === teacher.teacherId
    );
    
    if (!canEdit) {
      setErrorMessage('You are not assigned to teach this subject');
      setTimeout(() => setErrorMessage(''), 3000);
      return;
    }

    setEditingCell({ studentId, subjectId });
    setTempScore(currentScore || '');
  };

  // Save inline score
  const saveInlineScore = async (studentId, subjectId) => {
    if (tempScore === '' || tempScore === null || tempScore === undefined) {
      setEditingCell(null);
      return;
    }

    const scoreValue = parseInt(tempScore, 10);
    if (isNaN(scoreValue) || scoreValue < 0 || scoreValue > 100) {
      setErrorMessage('Score must be a number between 0-100');
      setTimeout(() => setErrorMessage(''), 3000);
      setEditingCell(null);
      return;
    }

    // Add to saving cells
    const cellKey = `${studentId}-${subjectId}`;
    setSavingCells(prev => new Set([...prev, cellKey]));

    try {
      // Check if score exists
      const existingScore = studentScoresData.find(
        s => s.studentId === studentId && s.subjectId === subjectId
      );

      console.log(existingScore)

      const url = existingScore 
        ? `http://127.0.0.1:8000/students/score/${existingScore.studentScoreId}`
        : `http://127.0.0.1:8000/students/score`;
      
      const method = existingScore ? 'PUT' : 'POST';

      const formData = new FormData();
      
      formData.append('score', scoreValue);
      formData.append('studentId', studentId);
      formData.append('subjectId', subjectId);

      const response = await fetch(url, {
        method,
        body: formData,
      });

      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      
      const data = await response.json();
      
      if (data?.status_code === 200) {
        // Update local state immediately
        if (existingScore) {
          setStudentScoresData(prev => 
            prev.map(score => 
              score.studentScoreId === existingScore.studentScoreId 
                ? { ...score, score: scoreValue }
                : score
            )
          );
        } else {
          // Add new score
          const newScore = {
            studentScoreId: data.data?.studentScoreId || Date.now(),
            studentId,
            subjectId,
            score: scoreValue,
            createdAt: new Date().toISOString()
          };
          setStudentScoresData(prev => [...prev, newScore]);
        }

        setSuccessMessage('Score updated successfully');
        setTimeout(() => setSuccessMessage(''), 2000);
      }
    } catch (error) {
      console.error('Error saving score:', error);
      setErrorMessage('Failed to save score');
      setTimeout(() => setErrorMessage(''), 3000);
    } finally {
      // Remove from saving cells
      setSavingCells(prev => {
        const newSet = new Set(prev);
        newSet.delete(cellKey);
        return newSet;
      });
      setEditingCell(null);
      setTempScore('');
    }
  };

  // Handle Enter key press
  const handleKeyPress = (e, studentId, subjectId) => {
    if (e.key === 'Enter') {
      saveInlineScore(studentId, subjectId);
    } else if (e.key === 'Escape') {
      setEditingCell(null);
      setTempScore('');
    }
  };

  // Handle click outside to save
  useEffect(() => {
    const handleClickOutside = (e) => {
      if (editingCell && !e.target.closest('.inline-score-cell')) {
        saveInlineScore(editingCell.studentId, editingCell.subjectId);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [editingCell, tempScore]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e) => {
      // Ctrl+S to save when editing
      if (e.ctrlKey && e.key === 's' && editingCell) {
        e.preventDefault();
        saveInlineScore(editingCell.studentId, editingCell.subjectId);
      }
      // Escape to cancel editing
      if (e.key === 'Escape' && editingCell) {
        setEditingCell(null);
        setTempScore('');
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [editingCell]);

  // STUDENT SELECTION FUNCTIONS

  // Check if all subjects are filled for a student
  const isAllSubjectsFilled = (studentId) => {
    const studentScores = studentScoresData.filter(score => score.studentId === studentId);
    return studentScores.length === subjects.length;
  };

  // Check if all subjects are signed for a student
  const isAllSubjectsSigned = (studentId) => {
    if (!Array.isArray(signatures)) return false;
    return teacherSubjects.every(subject => {
      return signatures.some(signature => 
        signature.subjectId === subject.subjectId && 
        signature.studentId === studentId
      );
    });
  };

  // Check if student is ready for confirmation
  const isReadyForConfirmation = (studentId) => {
    console.log(isAllSubjectsFilled(studentId), isAllSubjectsSigned(studentId));
    return isAllSubjectsFilled(studentId) && isAllSubjectsSigned(studentId);
  };

  // Handle select/deselect all
  const handleSelectAll = () => {
    if (selectAll) {
      setSelectedStudents(new Set());
    } else {
      // Only select students who are ready for confirmation
      const readyStudents = currentItems
        .map(student => String(student.studentId));
      
      const allReadyIds = new Set(readyStudents);
      setSelectedStudents(allReadyIds);
    }
    setSelectAll(!selectAll);
  };

  // Handle individual student selection
  const handleSelectStudent = (studentId) => {
    const newSelected = new Set(selectedStudents);
    if (newSelected.has(studentId)) {
      newSelected.delete(studentId);
    } else {
      newSelected.add(studentId);
    }
    setSelectedStudents(newSelected);
    
    // Update selectAll state
    if (newSelected.size === currentItems.length) {
      setSelectAll(true);
    } else if (selectAll) {
      setSelectAll(false);
    }
  };

  // Check if selected students can send reports
  const canSendSelectedReports = () => {
    if (selectedStudents.size === 0) return false;
    
    for (const studentId of selectedStudents) {
      if (!isReadyForConfirmation(Number(studentId))) {
        return false;
      }
    }
    return true;
  };

  // NOTIFICATION FUNCTIONS

  // Mark notification as seen
  const markNotificationAsSeen = (notificationId) => {
    setSeenNotifications(prev => {
      const newSet = new Set([...prev, notificationId]);
      // Save to localStorage
      localStorage.setItem('seenNotifications', JSON.stringify([...newSet]));
      return newSet;
    });
  };

  // Mark all notifications as read
  const markAllAsRead = () => {
    const allNotificationIds = unreadNotifications.map(notif => notif.reviewId);
    const newSeen = new Set([...seenNotifications, ...allNotificationIds]);
    setSeenNotifications(newSeen);
    localStorage.setItem('seenNotifications', JSON.stringify([...newSeen]));
  };

  // OTHER FUNCTIONS

  const getClassDisplayName = (classId) => {
    const cls = classes.find(c => c.classId === classId);
    if (!cls) return 'N/A';
    return `${cls.className}`;
  };

  // Export to Excel
  const exportToExcel = async () => {
    if (filteredStudents.length === 0) {
      setErrorMessage('No data to export');
      setTimeout(() => setErrorMessage(''), 3000);
      return;
    }

    try {
      const workbook = new ExcelJS.Workbook();
      const worksheet = workbook.addWorksheet('Student Scores');

      // Prepare headers
      const headers = ['Roll ID', 'Student Name', 'Class', 'Status'];
      const scoreHeaders = teacherSubjects.map(subject => 
        `${subject.subjectName} Score`
      );
      const signatureHeaders = teacherSubjects.map(subject =>
        `${subject.subjectName} Signature`
      );
      const allHeaders = [...headers, ...scoreHeaders, ...signatureHeaders];

      // Set column widths and add headers
      worksheet.columns = allHeaders.map(header => ({
        header,
        width: 25
      }));

      // Style headers
      const headerRow = worksheet.getRow(1);
      headerRow.eachCell((cell) => {
        cell.font = { bold: true, color: { argb: 'FFFFFFFF' }, name: 'Arial', size: 12 };
        cell.fill = {
          type: 'pattern',
          pattern: 'solid',
          fgColor: { argb: '1A47C2' },
        };
        cell.alignment = { horizontal: 'center', vertical: 'middle' };
        cell.border = {
          top: { style: 'medium' },
          left: { style: 'medium' },
          bottom: { style: 'medium' },
          right: { style: 'medium' }
        };
      });
      
      // Freezing header
      worksheet.views = [
        {
          state: 'frozen',
          xSplit: 0,
          ySplit: 1,
          activeCell: 'A2',
          showGridLines: true
        }
      ];

      // Add data rows
      filteredStudents.forEach((student, index) => {
        const classInfo = classes.find(c => c.classId === student.classId);
        
        const rowData = [
          student.rollId || `CLS${student.classId}000${student.studentId}`,
          student.studentName || 'N/A',
          classInfo?.className || 'N/A',
          student.active ? 'Active' : 'Inactive'
        ];

        // Add scores for each subject
        teacherSubjects.forEach(subject => {
          const score = studentScoresData.find(
            s => s.studentId === student.studentId && s.subjectId === subject.subjectId
          )?.score || 'N/A';
          rowData.push(score);
        });

        // Add signature status for each subject
        teacherSubjects.forEach(subject => {
          const hasSignature = signatures[subject.subjectId] ? 'Signed' : 'Not Signed';
          rowData.push(hasSignature);
        });

        const row = worksheet.addRow(rowData);
        
        // Style data rows
        row.eachCell((cell) => {
          cell.font = { size: 11 };
          cell.border = {
            top: { style: 'thin' },
            left: { style: 'thin' },
            bottom: { style: 'thin' },
            right: { style: 'thin' }
          };
          cell.alignment = { vertical: 'middle' };
        });

        // Alternate row colors
        if (index % 2 === 0) {
          row.eachCell((cell) => {
            cell.fill = {
              type: 'pattern',
              pattern: 'solid',
              fgColor: { argb: 'F3F4F6' }
            };
          });
        }
      });

      // Auto-fit columns
      worksheet.columns.forEach(column => {
        let maxLength = 0;
        column.eachCell({ includeEmpty: true }, cell => {
          const cellLength = cell.value ? cell.value.toString().length : 10;
          if (cellLength > maxLength) {
            maxLength = cellLength;
          }
        });
        column.width = Math.min(Math.max(maxLength + 2, 15), 30);
      });

      // Generate and download
      const buffer = await workbook.xlsx.writeBuffer();
      const blob = new Blob([buffer], { 
        type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' 
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `student_scores_${new Date().toISOString().split('T')[0]}.xlsx`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Error exporting to Excel:', error);
      setErrorMessage('Failed to export data');
      setTimeout(() => setErrorMessage(''), 3000);
    }
  };

  // Cleanup URLs on unmount
  useEffect(() => {
    return () => {
      reports.forEach(report => {
        if (report.report && report.report.startsWith('blob:')) {
          URL.revokeObjectURL(report.report);
        }
      });
    };
  }, [reports]);

  // SIGNATURE FUNCTIONS
  const startDrawing = (e) => {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const ctx = canvas.getContext('2d');
    ctx.beginPath();
    ctx.moveTo(e.clientX - rect.left, e.clientY - rect.top);
    setIsDrawing(true);
  };

  const draw = (e) => {
    if (!isDrawing) return;
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const ctx = canvas.getContext('2d');
    ctx.lineTo(e.clientX - rect.left, e.clientY - rect.top);
    ctx.strokeStyle = '#1e40af';
    ctx.lineWidth = 2;
    ctx.lineCap = 'round';
    ctx.stroke();
  };

  const stopDrawing = () => {
    setIsDrawing(false);
  };

  const clearSignature = () => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  };

  const dataURLToBlob = (dataURL) => {
    const arr = dataURL.split(',');
    const mime = arr[0].match(/:(.*?);/)[1];
    const bstr = atob(arr[1]);
    let n = bstr.length;
    const u8arr = new Uint8Array(n);
  
    while (n--) {
      u8arr[n] = bstr.charCodeAt(n);
    }
  
    return new Blob([u8arr], { type: mime });
  };

  const saveSignature = async () => {
    const canvas = canvasRef.current;
  
    // Get Data URL from canvas
    const signatureDataURL = canvas.toDataURL('image/png');
  
    // Convert Data URL to Blob
    const signatureBlob = dataURLToBlob(signatureDataURL);
  
    // Convert Blob to File
    const signatureFile = new File(
      [signatureBlob],
      'signature.png',
      { type: 'image/png' }
    );
  
    const formData = new FormData();
    formData.append('signature', signatureFile);
    formData.append('subjectId', signatureSubject.subjectId);
    formData.append('teacherId', teacher.teacherId);
    formData.append('studentId', Array.from(selectedStudents));
    formData.append(
      'score',
      studentScoresData.find(
        score => score.subjectId === signatureSubject.subjectId
      )?.score || ''
    );
  
    const saveSign = await fetch('http://127.0.0.1:8000/teachers/teacherSign', {
      method: 'POST',
      body: formData,
    });
  
    if (saveSign.ok) {
      const response = await saveSign.json();
      console.log(response);
  
      if (response?.status_code === 200) {
        setShowSignatureModal(false);
        setPopupMsg('Signature saved successfully');
  
        setTimeout(() => {
          setPopupMsg('');
          setShowSignatureModal(false);
          window.location.reload();
        }, 3000);
      }
    }
  };

  // REPORT FUNCTIONS
  const subjectScores = students.map((student) => {
    const studentSubjectScores = subjects.map((subject) => {
      const score = studentScoresData.find((score) => score.studentId === student.studentId && score.subjectId === subject.subjectId);
      if (score === undefined) {
        return {
          subjectName: subject.subjectName,
          score: 0,
          maxScore: 100,
          grade: grades.find((grade) => grade.gradeId === 6)?.gradeLetter
        };
      }
      return {
        subjectName: subject.subjectName,
        score: score ? score.score : 0,
        maxScore: 100,
        grade: grades.find((grade) => grade.gradeId === score?.grade)?.gradeLetter
      };
    });
    return {
      ...student,
      subjectScores: studentSubjectScores
    }
  });

  useEffect(() => {
    const handleSendReports = async () => {
      for (let i = 0; i < pdfData.allPdfs.length; i++) {
        const blob = pdfData.allPdfs[i].blob;
        const file = new File([blob], 'report.pdf', { type: 'application/pdf' });
        const formData = new FormData();
        formData.append('attachment', file);
        formData.append('studentId', pdfData.allPdfs[i].studentId);
        formData.append('teacherId', teacher.teacherId);
        formData.append('comments', comment || 'Student subject wise Report');
        const response = await fetch('http://127.0.0.1:8000/teachers/uploadReport', {
          method: 'POST',
          body: formData
        });
        const data = await response.json();
        console.log(data);
        if(data?.status_code === 200) {
          setPopupMsg('Report sent successfully');
          setTimeout(() => {
            setPopupMsg('');
            setShowSendReportModal(false);
          }, 5000);
        } else {
          setPopupMsg('Error sending report');
          setTimeout(() => setPopupMsg(''), 3000);
        }
      }
    }
    if(pdfData) {
      handleSendReports();
    }
  }, [pdfData]);

  const SendReport = async () => {
    setGeneratePdf(true);
  };

  const handlePdfGenerated = (pdfInfo) => {
    console.log("PDF generated successfully:", pdfInfo);
    // setPdfData(pdfInfo);
    setGeneratePdf(false);
  };

  if (loading) return <Loader />;

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-50">
      {/* Top Navigation Bar */}
      <div className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center space-x-8">
              <div>
                <h1 className="text-2xl font-bold text-gray-900">Class Teacher Dashboard</h1>
                <p className="text-sm text-gray-600">
                  Welcome back, <span className="font-semibold text-indigo-600">{user?.userName || 'Teacher'}</span>
                </p>
              </div>
            </div>
            <div className="flex items-center space-x-3">
              <button
                className="relative flex items-center mr-2"
                onClick={() => setOpenList(true)}
              >
                <Bell size={18} />
                {notificationCount > 0 && (
                  <span className="absolute -top-1 -right-1 bg-red-500 text-white text-xs rounded-full w-5 h-5 flex items-center justify-center">
                    {notificationCount}
                  </span>
                )}
              </button>

              {openList && (
                <div className="absolute right-1 top-1 mt-2 w-80 bg-white shadow-lg rounded-lg z-50">
                  <div className="w-full p-3 border-b flex justify-between items-center">
                    <h2 className="text-lg font-semibold">Notifications</h2>
                    <button onClick={() => setOpenList(false)} className='font-semibold'>
                      <X size={18} />
                    </button>
                  </div>

                  {unreadNotifications.length === 0 ? (
                    <div className="p-4 text-gray-500 text-sm">
                      No new notifications
                    </div>
                  ) : (
                    unreadNotifications.map((item) => (
                      <div
                        onClick={() => {
                          setSelectedNotification(item);
                          markNotificationAsSeen(item.reviewId);
                        }}
                        key={item.reviewId}
                        className="p-3 border-b hover:bg-gray-50 cursor-pointer bg-blue-50"
                      >
                        <div className="flex items-start justify-between">
                          <div className="flex-1">
                            <div className="flex items-center gap-2">
                              <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                              <p className="font-medium text-gray-900">
                                {reports.find(report => report.reportId === item.reportId)?.comments || 'New Comment'}
                              </p>
                            </div>
                            <p className="text-sm text-gray-600 mt-1 truncate">
                              {item.comments}
                            </p>
                            <div className="flex items-center justify-between mt-2">
                              <span className="text-xs text-gray-400">
                                {new Date(item.created_at).toLocaleDateString('en-US', {
                                  month: 'short',
                                  day: 'numeric',
                                  hour: '2-digit',
                                  minute: '2-digit'
                                })}
                              </span>
                              <span className="text-xs px-2 py-1 bg-blue-100 text-blue-800 rounded">
                                New
                              </span>
                            </div>
                          </div>
                        </div>
                      </div>
                    ))
                  )}

                  {unreadNotifications.length > 0 && (
                    <div className="p-3 border-t">
                      <button
                        onClick={() => {
                          markAllAsRead();
                          setOpenList(false);
                        }}
                        className="w-full text-center text-sm text-blue-600 hover:text-blue-800 font-medium"
                      >
                        Mark all as read
                      </button>
                    </div>
                  )}
                </div>
              )}

              {selectedNotification && (
                <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50">
                  <div className="bg-white rounded-lg w-96 p-5">
                    <h2 className="text-lg font-semibold">
                      {reports.find(r => r.reportId === selectedNotification.reportId)?.comments}
                    </h2>

                    <p className="mt-3 text-gray-700">
                      Student Name: {students.find(student => 
                        reports.find(report => report.reportId === selectedNotification.reportId && 
                        report.studentId === student.studentId)
                      )?.studentName}
                    </p>
                    
                    <p className="mt-3 text-gray-700">
                      Student Roll No: {students.find(student => 
                        reports.find(report => report.reportId === selectedNotification.reportId && 
                        report.studentId === student.studentId)
                      )?.rollId}
                    </p>

                    <p className="mt-3 text-gray-700">
                      Comments: {selectedNotification?.comments}
                    </p>

                    <p className="mt-2 text-xs text-gray-400">
                      {selectedNotification.created_at}
                    </p>

                    <button
                      className="mt-4 px-4 py-2 bg-blue-600 text-white rounded"
                      onClick={() => setSelectedNotification(null)}
                    >
                      Close
                    </button>
                  </div>
                </div>
              )}

              <button
                onClick={exportToExcel}
                className="flex items-center gap-2 bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded-lg transition-colors font-medium disabled:opacity-50 disabled:cursor-not-allowed"
                disabled={filteredStudents.length === 0}
              >
                <Download size={18} />
                Export
              </button>
              <button
                onClick={() => dispatch(logout())}
                className="flex items-center gap-2 bg-red-500 hover:bg-red-600 text-white px-4 py-2 rounded-lg transition-colors font-medium"
              >
                <LogOut size={18} />
                Logout
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {/* Error and Success Messages */}
        {errorMessage && (
          <div className="mb-4 p-4 bg-red-50 border border-red-200 text-red-700 rounded-lg">
            {errorMessage}
          </div>
        )}
        
        {successMessage && (
          <div className="mb-4 p-4 bg-green-50 border border-green-200 text-green-700 rounded-lg">
            {successMessage}
          </div>
        )}
        
        {popUpMsg && (
          <div className="mb-4 p-4 bg-blue-50 border border-blue-200 text-blue-700 rounded-lg">
            {popUpMsg}
          </div>
        )}

        {/* Teacher Info Card */}
        <div className="bg-white rounded-xl shadow-md p-6 mb-6">
          <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
            <div className="flex-1">
              <h2 className="text-lg font-semibold text-gray-800 mb-2">Teaching Profile</h2>
              <div className="flex flex-wrap gap-2 mb-3">
                <span className="inline-flex items-center gap-1 bg-blue-100 text-blue-800 px-3 py-1 rounded-full text-sm">
                  <GraduationCap size={14} />
                  Subjects: {teacherSubjects.length}
                </span>
                <span className="inline-flex items-center gap-1 bg-purple-100 text-purple-800 px-3 py-1 rounded-full text-sm">
                  <Users size={14} />
                  Classes: {teacherClasses.length}
                </span>
                <span className="inline-flex items-center gap-1 bg-green-100 text-green-800 px-3 py-1 rounded-full text-sm">
                  <Award size={14} />
                  Total Students: {quickStats.totalStudents}
                </span>
              </div>
              <div className="flex flex-wrap gap-2">
                {teacherSubjects.length > 0 ? (
                  teacherSubjects.map(subject => (
                    <span 
                      key={subject.subjectId} 
                      className={`px-2 py-1 rounded text-sm cursor-pointer hover:opacity-80 transition ${'bg-indigo-50 text-indigo-700'}`}
                      onClick={() => {
                        setSignatureSubject(subject);
                      }}
                    >
                      {subject.subjectName}
                    </span>
                  ))
                ) : (
                  <span className="text-gray-500 text-sm">No subjects assigned</span>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Tab Navigation */}
        <div className="flex gap-3 mb-6">
          <button
            onClick={() => setActiveTab('dashboard')}
            className={`px-4 py-2 rounded-lg font-medium transition-colors text-white flex items-center gap-2 ${
              activeTab === 'dashboard' ? 'bg-blue-500' : 'bg-gray-500 hover:bg-gray-600'
            }`}
          >
            <Home size={20} />
            Dashboard
          </button>
          <button
            onClick={() => setActiveTab('reports')}
            className={`px-4 py-2 rounded-lg font-medium transition-colors text-white flex items-center gap-2 ${
              activeTab === 'reports' ? 'bg-blue-500' : 'bg-gray-500 hover:bg-gray-600'
            }`}
          >
            <FileText size={20} />
            Reports ({reports.length})
          </button>
        </div>

        {/* Dashboard Content */}
        {activeTab === 'dashboard' && (
          <>
            {/* Quick Stats */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-6 mb-6">
              <div className="bg-white rounded-xl shadow-sm p-6">
                <div className="flex items-center gap-3">
                  <div className="w-12 h-12 bg-blue-100 rounded-full flex items-center justify-center">
                    <Users className="text-blue-600" size={24} />
                  </div>
                  <div>
                    <p className="text-sm text-gray-600">Total Students</p>
                    <p className="text-2xl font-bold text-gray-900">{quickStats.totalStudents}</p>
                  </div>
                </div>
              </div>
              
              <div className="bg-white rounded-xl shadow-sm p-6">
                <div className="flex items-center gap-3">
                  <div className="w-12 h-12 bg-green-100 rounded-full flex items-center justify-center">
                    <BarChart3 className="text-green-600" size={24} />
                  </div>
                  <div>
                    <p className="text-sm text-gray-600">Average Score</p>
                    <p className="text-2xl font-bold text-gray-900">{quickStats.averageScore}%</p>
                  </div>
                </div>
              </div>
              
              <div className="bg-white rounded-xl shadow-sm p-6">
                <div className="flex items-center gap-3">
                  <div className="w-12 h-12 bg-yellow-100 rounded-full flex items-center justify-center">
                    <Award className="text-yellow-600" size={24} />
                  </div>
                  <div>
                    <p className="text-sm text-gray-600">Top Performers</p>
                    <p className="text-2xl font-bold text-gray-900">{quickStats.topPerformers}</p>
                  </div>
                </div>
              </div>
              
              <div className="bg-white rounded-xl shadow-sm p-6">
                <div className="flex items-center gap-3">
                  <div className="w-12 h-12 bg-orange-100 rounded-full flex items-center justify-center">
                    <AlertCircle className="text-orange-600" size={24} />
                  </div>
                  <div>
                    <p className="text-sm text-gray-600">Pending Scores</p>
                    <p className="text-2xl font-bold text-gray-900">{quickStats.pendingScores}</p>
                  </div>
                </div>
              </div>
              
              <div className="bg-white rounded-xl shadow-sm p-6">
                <div className="flex items-center gap-3">
                  <div className="w-12 h-12 bg-purple-100 rounded-full flex items-center justify-center">
                    <FileText className="text-purple-600" size={24} />
                  </div>
                  <div>
                    <p className="text-sm text-gray-600">Recent Submissions</p>
                    <p className="text-2xl font-bold text-gray-900">{quickStats.recentSubmissions}</p>
                  </div>
                </div>
              </div>
            </div>

            {/* Performance Trends */}
            {performanceTrend.length > 0 && (
              <div className="bg-white rounded-xl shadow-sm p-6 mb-6">
                <h3 className="text-lg font-semibold text-gray-800 mb-4">Subject Performance Trends</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {performanceTrend.map((trend, index) => (
                    <div key={index} className="border border-gray-200 rounded-lg p-4">
                      <div className="flex justify-between items-center mb-2">
                        <span className="font-medium text-gray-900">{trend.subject}</span>
                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                          trend.trend === 'up' ? 'bg-green-100 text-green-800' :
                          trend.trend === 'stable' ? 'bg-yellow-100 text-yellow-800' :
                          'bg-red-100 text-red-800'
                        }`}>
                          {trend.trend === 'up' ? '↑ Improving' : 
                           trend.trend === 'stable' ? '→ Stable' : '↓ Needs Attention'}
                        </span>
                      </div>
                      <div className="text-2xl font-bold text-gray-900">{trend.averageScore}%</div>
                      <div className="text-sm text-gray-600">{trend.totalStudents} students</div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Students Table Section */}
            <div className="bg-white rounded-xl shadow-sm overflow-hidden">
              {/* Filters */}
              <div className="p-6 border-b border-gray-200">
                <div className="flex flex-col md:flex-row gap-4 mb-4">
                  <div className="relative flex-1">
                    <Search className="absolute left-3 top-3 text-gray-400" size={20} />
                    <input
                      type="text"
                      placeholder="Search students by name..."
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      className="w-full pl-10 pr-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                    />
                  </div>
                  <div className="flex gap-2">
                    <select
                      value={selectedClass}
                      onChange={(e) => setSelectedClass(e.target.value)}
                      className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                    >
                      <option value="All">All Classes</option>
                      {teacherClasses.map(classItem => (
                        <option key={classItem.classId} value={classItem.classId}>
                          {classItem.className}
                        </option>
                      ))}
                    </select>
                    <select
                      value={selectedSubject}
                      onChange={(e) => setSelectedSubject(e.target.value)}
                      className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                    >
                      <option value="All">All Subjects</option>
                      {teacherSubjects.map(subject => (
                        <option key={subject.subjectId} value={subject.subjectId}>
                          {subject.subjectName}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
                <div className="flex items-center justify-between text-sm text-gray-600">
                  <div className="flex items-center justify-between gap-4">
                    <button
                      onClick={handleSelectAll}
                      className="flex items-center gap-2 text-indigo-600 hover:text-indigo-800 font-medium"
                    >
                      {selectAll ? <CheckSquare size={16} /> : <Square size={16} />}
                      {selectAll ? 'Deselect All' : 'Select All Ready'}
                    </button>
                    <span>Selected: {selectedStudents.size} | Showing {currentItems.length} of {filteredStudents.length}</span>
                  </div>
                  <div>
                    {selectedStudents.size > 0 && canSendSelectedReports() ? (
                      <button
                        onClick={() => setShowSendReportModal(true)}
                        className="flex items-center gap-2 bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded-lg transition-colors font-medium"
                      >
                        <SendHorizonal size={18} />
                        Send {selectedStudents.size} Report(s) to Head Teacher
                      </button>
                    ) : (
                      <div className="text-sm text-gray-500">
                        {selectedStudents.size > 0 
                          ? "Complete all scores and signatures for selected students"
                          : "Select students to send reports"}
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {/* Table */}
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-semibold text-gray-600 uppercase">Select</th>
                      <th className="px-6 py-3 text-left text-xs font-semibold text-gray-600 uppercase">Student</th>
                      <th className="px-6 py-3 text-left text-xs font-semibold text-gray-600 uppercase">Class</th>
                      {subjects.map(subject => (
                        <th key={subject.subjectId} className="px-6 py-3 text-left text-xs font-semibold text-gray-600 uppercase">
                          <div className="flex items-center justify-between gap-2">
                            <span>{subject.subjectName}</span>
                            <button
                              onClick={() => {
                                setSignatureSubject(subject);
                                (Array.from(selectedStudents)).length <= 0 ? alert('Please select at least one student') : setShowSignatureModal(true);
                              }}
                              className={`p-1 rounded hover:bg-gray-200 text-gray-400`}
                              title={signatures[subject.subjectId] ? 'Signed' : 'Not Signed'}
                            >
                              <Signature size={16} />
                            </button>
                          </div>
                        </th>
                      ))}
                      <th className="px-6 py-3 text-left text-xs font-semibold text-gray-600 uppercase">Status</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-200">
                    {currentItems.length > 0 ? (
                      currentItems.map((student) => {
                        const isReady = isReadyForConfirmation(student.studentId);
                        const allFilled = isAllSubjectsFilled(student.studentId);
                        const allSigned = isAllSubjectsSigned(student.studentId);

                        console.log(isReady);
                        
                        return (
                          <tr key={student.studentId} className="hover:bg-gray-50 transition-colors">
                            <td className="px-6 py-4">
                              <button
                                onClick={() => handleSelectStudent(String(student.studentId))}
                                className={`p-1 rounded ${isReady ? 'hover:bg-gray-200' : 'opacity-40 '}`}
                                title={!isReady ? "Complete all scores and signatures first" : "Select for sending report"}
                              >
                                {selectedStudents.has(String(student.studentId)) ? (
                                  <CheckSquare size={20} className="text-indigo-600" />
                                ) : (
                                  <Square size={20} className="text-gray-400" />
                                )}
                              </button>
                            </td>
                            <td className="px-6 py-4">
                              <div className="flex items-center gap-3">
                                <div className="w-8 h-8 bg-indigo-100 rounded-full flex items-center justify-center">
                                  <Users className="text-indigo-600" size={16} />
                                </div>
                                <div>
                                  <div className="font-medium text-gray-900">{student.studentName}</div>
                                  <div className="text-sm text-gray-500">Roll: {student.rollId}</div>
                                </div>
                              </div>
                            </td>
                            <td className="px-6 py-4 text-sm text-gray-900">
                              {getClassDisplayName(student.classId)}
                            </td>
                            {subjects.map(subject => {
                              const score = studentScoresData.find(
                                s => s.studentId === student.studentId && s.subjectId === subject.subjectId
                              );
                              const isSigned = Array.isArray(signatures) && signatures.some(signature => 
                                signature.subjectId === subject.subjectId && 
                                signature.studentId === student.studentId
                              );
                              const canEdit = mapTeacher.find(t => 
                                t.subjectId === subject.subjectId && t.teacherId === teacher.teacherId
                              );
                              const cellKey = `${student.studentId}-${subject.subjectId}`;
                              const isSaving = savingCells.has(cellKey);
                              const isEditing = editingCell?.studentId === student.studentId && 
                                                editingCell?.subjectId === subject.subjectId;
                              
                              return (
                                <td 
                                  key={subject.subjectId} 
                                  className="px-6 py-4 inline-score-cell"
                                >
                                  <div className="flex flex-col gap-2">
                                    <div className="flex items-center justify-between gap-2 min-h-[40px]">
                                      {isSaving ? (
                                        <div className="px-3 py-1 rounded bg-blue-50 min-w-[60px] flex justify-center items-center">
                                          <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
                                        </div>
                                      ) : isEditing ? (
                                        <div className="relative">
                                          <input
                                            type="number"
                                            min="0"
                                            max="100"
                                            value={tempScore}
                                            onChange={(e) => setTempScore(e.target.value)}
                                            onKeyDown={(e) => handleKeyPress(e, student.studentId, subject.subjectId)}
                                            className="w-20 px-2 py-1 border border-blue-300 rounded focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-center"
                                            autoFocus
                                            onFocus={(e) => e.target.select()}
                                          />
                                          <div className="text-xs text-gray-500 mt-1">Press Enter to save</div>
                                        </div>
                                      ) : score ? (
                                        <div 
                                          onClick={() => canEdit && handleInlineScoreUpdate(student.studentId, subject.subjectId, score.score)}
                                          className={`px-3 py-1 rounded text-center text-sm font-medium min-w-[60px] cursor-pointer transition-all hover:scale-105 ${
                                            score.score >= 80 ? 'bg-green-100 text-green-800 hover:bg-green-200' :
                                            score.score >= 60 ? 'bg-yellow-100 text-yellow-800 hover:bg-yellow-200' :
                                            'bg-red-100 text-red-800 hover:bg-red-200'
                                          } ${canEdit ? 'cursor-pointer' : 'cursor-default'}`}
                                          title={canEdit ? "Click to edit score" : "You cannot edit this subject"}
                                        >
                                          {score.score}%
                                        </div>
                                      ) : (
                                        <div 
                                          onClick={() => canEdit && handleInlineScoreUpdate(student.studentId, subject.subjectId)}
                                          className={`px-3 py-1 rounded text-center text-sm font-medium min-w-[60px] cursor-pointer transition-all hover:scale-105 ${
                                            canEdit 
                                              ? 'bg-gray-100 text-gray-600 hover:bg-gray-200 border border-dashed border-gray-300' 
                                              : 'bg-gray-50 text-gray-400'
                                          }`}
                                          title={canEdit ? "Click to add score" : "You cannot add score for this subject"}
                                        >
                                          {canEdit ? 'Add Score' : 'N/A'}
                                        </div>
                                      )}
                                    </div>
                                    {score && isSigned ? (
                                      <div className="text-xs text-green-600 flex items-center gap-1">
                                        <Check size={10} />
                                        Signed
                                      </div>
                                    ) : score && !isSigned ? (
                                      <div className="text-xs text-orange-600 flex items-center gap-1">
                                        <AlertCircle size={10} />
                                        Not Signed
                                      </div>
                                    ) : null}
                                  </div>
                                </td>
                              );
                            })}
                            <td className="px-6 py-4">
                              <div className={`text-xs px-3 py-1 rounded text-center font-medium ${
                                allFilled && allSigned
                                  ? 'bg-green-100 text-green-800' 
                                  : allFilled && !allSigned
                                    ? 'bg-yellow-100 text-yellow-800'
                                    : 'bg-gray-100 text-gray-800'
                              }`}>
                                {allFilled && allSigned ? 'Ready' : 
                                 allFilled ? 'Scores Complete' : 
                                 'Incomplete'}
                              </div>
                              <div className="text-xs text-gray-500 mt-1">
                                Scores: {studentScoresData.filter(s => s.studentId === student.studentId).length}/{subjects.length}
                                <br />
                                Signed: {Array.isArray(signatures) ? signatures.filter(s => s.studentId === student.studentId).length : 0}/{subjects.length}
                              </div>
                            </td>
                          </tr>
                        );
                      })
                    ) : (
                      <tr>
                        <td colSpan={4 + teacherSubjects.length} className="px-6 py-8 text-center text-gray-500">
                          No students match your filters
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>

              {/* Pagination */}
              {currentItems.length > 0 && (
                <div className="px-6 py-4 border-t border-gray-200 bg-gray-50">
                  <div className="flex flex-col sm:flex-row justify-between items-center gap-4">
                    <div className="flex items-center gap-4">
                      <div className="flex items-center gap-2">
                        <span className="text-sm text-gray-600">Show:</span>
                        <select
                          value={itemsPerPage}
                          onChange={(e) => {
                            setItemsPerPage(Number(e.target.value));
                            setCurrentPage(1);
                          }}
                          className="text-sm border border-gray-300 rounded px-2 py-1 bg-white"
                        >
                          <option value="5">5</option>
                          <option value="10">10</option>
                          <option value="25">25</option>
                          <option value="50">50</option>
                        </select>
                        <span className="text-sm text-gray-600">per page</span>
                      </div>
                      <div className="text-sm text-gray-600">
                        Page {currentPage} of {totalPages}
                      </div>
                    </div>
                    
                    <div className="flex items-center gap-2">
                      <button
                        onClick={() => handlePageChange(currentPage - 1)}
                        disabled={currentPage === 1}
                        className={`px-3 py-1 rounded border ${currentPage === 1 ? 'text-gray-400 border-gray-300 cursor-not-allowed' : 'text-gray-700 border-gray-300 hover:bg-gray-100'}`}
                      >
                        <ChevronLeft size={16} />
                      </button>
                      
                      {getPageNumbers().map((page, index) => (
                        <button
                          key={index}
                          onClick={() => typeof page === 'number' ? handlePageChange(page) : null}
                          className={`px-3 py-1 rounded border ${
                            currentPage === page 
                              ? 'bg-indigo-600 text-white border-indigo-600' 
                              : 'text-gray-700 border-gray-300 hover:bg-gray-100'
                          } ${typeof page !== 'number' ? 'cursor-default hover:bg-transparent' : ''}`}
                          disabled={typeof page !== 'number'}
                        >
                          {page}
                        </button>
                      ))}
                      
                      <button
                        onClick={() => handlePageChange(currentPage + 1)}
                        disabled={currentPage === totalPages}
                        className={`px-3 py-1 rounded border ${currentPage === totalPages ? 'text-gray-400 border-gray-300 cursor-not-allowed' : 'text-gray-700 border-gray-300 hover:bg-gray-100'}`}
                      >
                        <ChevronRight size={16} />
                      </button>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </>
        )}

        {/* Reports Tab */}
        {activeTab === 'reports' && (
          <div className="bg-white rounded-xl shadow-sm overflow-hidden">
            <div className="p-6 border-b border-gray-200">
              <div className="flex justify-between items-center">
                <h3 className="text-lg font-semibold text-gray-800">Submitted Reports</h3>
                <button
                  onClick={() => setShowUploadModal(true)}
                  className="flex items-center gap-2 bg-indigo-600 hover:bg-indigo-700 text-white px-4 py-2 rounded-lg transition-colors"
                >
                  <Upload size={18} />
                  Upload Report
                </button>
              </div>
            </div>
            <div className="p-6">
              {reports.length > 0 ? (
                reports.map(report => (
                  <div
                    key={report.reviewId || report.id}
                    onClick={() => setSelectedReport(report)}
                    className="flex items-center bg-white mb-3 justify-between p-4 border border-gray-200 rounded-lg hover:bg-gray-50 cursor-pointer transition"
                  >
                    <div className="flex-1">
                      <h3 className="font-semibold text-gray-800">
                        {report.comments || 'Report'}
                      </h3>
                      <p className="text-sm text-gray-600">
                        Teacher: {report.teacherId === teacher.teacherId ? teacher.teacherName : 'Unknown'}
                      </p>
                      <p className="text-sm text-gray-500">
                        Submitted: {new Date(report.createdAt || Date.now()).toLocaleDateString()}
                      </p>
                      <p className={`text-sm text-gray-500 flex items-center`}>
                        Status: <p className={`font-bold ${report.status ? 'text-green-600' : 'text-red-600'}`}>{report.status ? 'Approved' : 'Pending'}</p>
                      </p>
                    </div>
                    <div className="flex items-center gap-3">
                      {report.comments && (
                        <MessageSquare size={20} className="text-blue-600" />
                      )}
                    </div>
                  </div>
                ))
              ) : (
                <div className="text-center py-8 text-gray-500">
                  <FileText size={48} className="mx-auto mb-4 text-gray-300" />
                  <p className="text-lg font-medium mb-2">No reports submitted yet</p>
                  <p className="text-gray-600">Upload your first report to get started</p>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Send Report Modal */}
        {showSendReportModal && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
            <div className="bg-white rounded-xl shadow-2xl max-w-md w-full">
              <div className="p-6 border-b border-gray-200">
                <div className="flex justify-between items-center">
                  <h3 className="text-lg font-semibold text-gray-900">Send Reports to Head Teacher</h3>
                  <button
                    onClick={() => setShowSendReportModal(false)}
                    className="text-gray-400 hover:text-gray-600 transition p-1 rounded-full hover:bg-gray-100"
                  >
                    <X size={24} />
                  </button>
                </div>
              </div>
              <div className="p-6">
                <div className="mb-6">
                  {popUpMsg && <p className="text-sm text-green-600 mt-1 mb-1">{popUpMsg}</p>}
                  <div className="flex items-center gap-2 mb-4">
                    <SendHorizonal className="text-indigo-600" size={24} />
                    <p className="font-medium text-gray-900">
                      Sending {selectedStudents.size} report(s) to head teacher
                    </p>
                  </div>
                  
                  <div className="bg-blue-50 p-4 rounded-lg mb-4">
                    <p className="text-sm text-blue-800 mb-2">
                      <strong>Note:</strong> Reports will include:
                    </p>
                    <ul className="text-sm text-blue-800 space-y-1">
                      <li className="flex items-center gap-2">
                        <Check className="text-green-600" size={16} />
                        All subject scores
                      </li>
                      <li className="flex items-center gap-2">
                        <Check className="text-green-600" size={16} />
                        Teacher signatures
                      </li>
                      <li className="flex items-center gap-2">
                        <Check className="text-green-600" size={16} />
                        Student information
                      </li>
                    </ul>
                  </div>
                </div>
                <div className="flex gap-3">
                  <button
                    onClick={() => setShowSendReportModal(false)}
                    className="flex-1 px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition"
                    disabled={sendingReports}
                  >
                    Cancel
                  </button>
                  <button
                    onClick={SendReport}
                    disabled={sendingReports}
                    className={`flex-1 px-4 py-2 rounded-lg transition flex items-center justify-center gap-2 ${
                      sendingReports
                        ? 'bg-gray-400 cursor-not-allowed'
                        : 'bg-indigo-600 hover:bg-indigo-700 text-white'
                    }`}
                  >
                    {sendingReports ? (
                      <>
                        <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                        Sending...
                      </>
                    ) : (
                      <>
                        <SendHorizonal size={18} />
                        Send Reports
                      </>
                    )}
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Signature Modal */}
        {showSignatureModal && (
          <Modal
            title="Add Signature"
            onClose={() => setShowSignatureModal(false)}
          >
            <div className="space-y-6">
              <div>
                <h3 className="font-semibold text-gray-900 mb-3">Draw Signature</h3>
                <div className="border-2 border-gray-300 rounded-xl overflow-hidden bg-gray-50">
                  <canvas
                    ref={canvasRef}
                    width={450}
                    height={200}
                    className="w-full cursor-crosshair"
                    onMouseDown={startDrawing}
                    onMouseMove={draw}
                    onMouseUp={stopDrawing}
                    onMouseLeave={stopDrawing}
                  />
                </div>
                <button
                  onClick={clearSignature}
                  className="mt-2 text-sm text-red-600 hover:text-red-700 font-medium"
                >
                  Clear Canvas
                </button>
              </div>

              <div>
                <h3 className="font-semibold text-gray-900 mb-3">Or Upload Image</h3>
                <label className="flex items-center justify-center gap-2 border-2 border-dashed border-gray-300 rounded-xl p-6 cursor-pointer hover:border-blue-500 transition bg-gray-50">
                  <Upload size={20} className="text-gray-400" />
                  <span className="text-gray-600">Upload signature image</span>
                  <input
                    type="file"
                    accept="image/*"
                    onChange={(e) => {
                      const file = e.target.files[0];
                      if (file) {
                        const reader = new FileReader();
                        reader.onload = async () => {
                          const signatureData = reader.result;
                          const formData = new FormData();
                          formData.append('signature', file);
                          formData.append('subjectId', signatureSubject.subjectId);
                          formData.append('teacherId', teacher.teacherId);
                          formData.append('studentId', Array.from(selectedStudents));
                          formData.append('score', studentScoresData.find(score => score.subjectId === signatureSubject.subjectId)?.score || '');
                          const saveSign = await fetch('http://127.0.0.1:8000/teachers/teacherSign', {
                            method: 'POST',
                            body: formData,
                          })
                          if (saveSign.ok) {
                            const response = await saveSign.json();
                            console.log(response);
                            if(response?.status_code === 200) {
                              setShowSignatureModal(false);
                              setPopupMsg('Signature saved successfully');
                              setTimeout(() => {
                                setPopupMsg('');
                                setShowSignatureModal(false);
                                window.location.reload();
                              }, 3000);
                            }
                          }
                        }
                        reader.readAsDataURL(file);
                      }
                    }}
                    className="hidden"
                  />
                </label>
              </div>

              <div className="flex gap-3 justify-end">
                <button
                  onClick={() => setShowSignatureModal(false)}
                  className="px-6 py-2 border border-gray-300 rounded-lg hover:bg-gray-100 transition"
                >
                  Cancel
                </button>
                <button
                  onClick={saveSignature}
                  className="flex items-center gap-2 bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 transition"
                >
                  <Check size={20} />
                  Save Signature
                </button>
              </div>
            </div>
          </Modal>
        )}

        {/* PDF Generation */}
        {generatePdf && (
          <BatchStudentReportCard
            schoolDetail={school}
            studentsData={subjectScores.filter(student => (Array.from(selectedStudents)).includes(student.studentId.toString()))}
            onBatchComplete={handlePdfGenerated}
          />
        )}

        {/* Upload Report Modal */}
        {showUploadModal && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
            <div className="bg-white rounded-xl shadow-2xl max-w-md w-full">
              <div className="p-6 border-b border-gray-200">
                <div className="flex justify-between items-center">
                  <h3 className="text-lg font-semibold text-gray-900">Upload Report</h3>
                  <button
                    onClick={() => {
                      setShowUploadModal(false);
                      setUploadedReport(null);
                      setComment('');
                    }}
                    className="text-gray-400 hover:text-gray-600 transition p-1 rounded-full hover:bg-gray-100"
                  >
                    <X size={24} />
                  </button>
                </div>
              </div>
              <div className="p-6">
                <div className="mb-6">
                  <label className="block text-sm font-medium text-gray-700 mb-2">Report Comments</label>
                  <textarea
                    value={comment}
                    onChange={(e) => setComment(e.target.value)}
                    placeholder="Enter report description or comments..."
                    className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent mb-4"
                    rows="3"
                  />
                  
                  <label className="block text-sm font-medium text-gray-700 mb-2">Select Report File</label>
                  <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-indigo-500 transition-colors cursor-pointer">
                    <input
                      type="file"
                      accept=".pdf,.doc,.docx"
                      onChange={(e) => {
                        const file = e.target.files[0];
                        if (file && file.size > 10 * 1024 * 1024) {
                          setErrorMessage('File size must be less than 10MB');
                          return;
                        }
                        setUploadedReport(file);
                        setErrorMessage('');
                      }}
                      className="hidden"
                      id="report-upload"
                    />
                    <label htmlFor="report-upload" className="cursor-pointer">
                      <Upload className="mx-auto text-gray-400 mb-3" size={40} />
                      <p className="text-gray-600 mb-2">Click to upload or drag and drop</p>
                      <p className="text-sm text-gray-500">PDF, DOC, DOCX (max 10MB)</p>
                    </label>
                  </div>
                  {uploadedReport && (
                    <div className="mt-3 p-3 bg-green-50 rounded-lg flex justify-between items-center">
                      <p className="text-green-700 truncate">{uploadedReport.name}</p>
                      <button
                        onClick={() => setUploadedReport(null)}
                        className="text-red-500 hover:text-red-700"
                      >
                        <X size={18} />
                      </button>
                    </div>
                  )}
                </div>
                <div className="flex gap-3">
                  <button
                    onClick={() => {
                      setShowUploadModal(false);
                      setUploadedReport(null);
                      setComment('');
                    }}
                    className="flex-1 px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={async () => {
                      if (!uploadedReport) {
                        setErrorMessage('Please select a file to upload');
                        return;
                      }
                
                      if (!teacher?.teacherId) {
                        setErrorMessage('Teacher information not available');
                        return;
                      }
                
                      const formData = new FormData();
                      formData.append('attachment', uploadedReport);
                      formData.append('teacherId', teacher.teacherId);
                      formData.append('comments', comment || 'Student subject wise Report');
                
                      try {
                        const response = await fetch('http://127.0.0.1:8000/teachers/uploadReport', {
                          method: 'POST',
                          body: formData
                        });
                        
                        if (!response.ok) {
                          throw new Error(`HTTP error! status: ${response.status}`);
                        }
                        
                        const data = await response.json();
                        setPopupMsg(data.message || 'Report uploaded successfully');
                        
                        setTimeout(() => {
                          setUploadedReport(null);
                          setShowUploadModal(false);
                          setPopupMsg('');
                          setComment('');
                        }, 3000);
                        
                      } catch (error) {
                        console.error('Error uploading report:', error);
                        setErrorMessage('Failed to upload report');
                        setTimeout(() => setErrorMessage(''), 3000);
                      }
                    }}
                    disabled={!uploadedReport}
                    className={`flex-1 px-4 py-2 rounded-lg transition ${
                      uploadedReport
                        ? 'bg-indigo-600 text-white hover:bg-indigo-700'
                        : 'bg-gray-200 text-gray-500 cursor-not-allowed'
                    }`}
                  >
                    Upload Report
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

const Modal = ({ title, children, onClose, width = 'max-w-lg' }) => (
  <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
    <div className={`bg-white rounded-2xl shadow-2xl ${width} w-full max-h-[90vh] overflow-hidden`}>
      <div className="bg-gradient-to-r from-indigo-600 to-purple-600 text-white p-6 flex items-center justify-between">
        <h2 className="text-2xl font-bold">{title}</h2>
        <button
          onClick={onClose}
          className="text-white hover:text-gray-200 text-2xl"
        >
          ✕
        </button>
      </div>
      <div className="p-6 overflow-y-auto max-h-[calc(90vh-80px)]">
        {children}
      </div>
    </div>
  </div>
);

export default ClassTeacherDashboard;