import React, { useState, useRef, useEffect } from 'react';
import { 
  Download, Upload, Edit3, X, Check, Phone, Mail, Globe, 
  XCircle, CheckCircle, ZoomIn, ZoomOut, MessageSquare, 
  Send, FileText, Save, User, Calendar, ChevronRight 
} from 'lucide-react';
import logo from '../assets/logo.png';
import MyPdfViewer from './pdfViewer';
import { useDispatch, useSelector } from 'react-redux';
import { useParams } from 'react-router-dom';
import Loader from './loader';
import html2canvas from 'html2canvas';
import jsPDF from 'jspdf';
import { initializeAuth } from '../authSlice';

const StudentReportCard = () => {
  const { user } = useSelector((state) => state.auth);
  const { id } = useParams();

  const dispatch = useDispatch();

  useEffect(() => {
    // Initialize auth from localStorage on app load
    dispatch(initializeAuth());
  }, [dispatch]);

  const backend_url = process.env.REACT_APP_BACKEND_URL;
  
  // State Management
  const [signature, setSignature] = useState(null);
  const [showSignatureModal, setShowSignatureModal] = useState(false);
  const [popUpMsg, setPopupMsg] = useState('');
  const [isDrawing, setIsDrawing] = useState(false);
  const canvasRef = useRef(null);
  const reportRef = useRef(null);
  const [reports, setReports] = useState([]);
  const [loading, setLoading] = useState(true);
  const [studentData, setStudentData] = useState({});
  
  // Report viewing states
  const [showReportsList, setShowReportsList] = useState(false);
  const [selectedReport, setSelectedReport] = useState(null);
  const [zoomLevel, setZoomLevel] = useState(100);
  const [comment, setComment] = useState('');
  const [comments, setComments] = useState([]);
  const [newComment, setNewComment] = useState('');

  // Drawing functions
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

  const saveSignature = () => {
    const canvas = canvasRef.current;
    const signatureData = canvas.toDataURL();
    setSignature(signatureData);
    setShowSignatureModal(false);
    setPopupMsg('Signature saved successfully');
    setTimeout(() => setPopupMsg(''), 3000);
  };

  const handleSignatureUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        setSignature(event.target.result);
        setPopupMsg('Signature uploaded successfully');
        setTimeout(() => setPopupMsg(''), 3000);
      };
      reader.readAsDataURL(file);
    }
  };

  // PDF Generation and Upload
  const createPdfBlob = async () => {
    const input = reportRef.current;
    const canvas = await html2canvas(input, {
      scale: 2,
      useCORS: true,
    });

    const imgData = canvas.toDataURL("image/png");
    const pdf = new jsPDF("p", "mm", "a4");
    const width = pdf.internal.pageSize.getWidth();
    const height = (canvas.height * width) / canvas.width;

    pdf.internal.pageSize.setHeight(height);
    pdf.internal.pageSize.setWidth(width);
    pdf.addImage(imgData, "PNG", 0, 0, width, height);

    if (pdf.internal.pageSize.getHeight() < height) {
      pdf.addPage();
      pdf.addImage(imgData, "PNG", 0, 0, width, height);
    }

    return pdf.output("blob");
  };

  const handleReportClick = (report) => {
          console.log(report);
          setSelectedReport(report);
          setZoomLevel(100);
          setComment(report.comments || '');
        };

  const handleUploadReport = async () => {
    if (!signature) {
      setPopupMsg('Please add signature first');
      setTimeout(() => setPopupMsg(''), 3000);
      return;
    }

    const pdfBlob = await createPdfBlob();
    const file = new File([pdfBlob], "report.pdf", { type: "application/pdf" });
    const formData = new FormData();
    formData.append('file', file);
    formData.append('student_id', id);
    formData.append('comments', comment || 'Student Final Report');
    formData.append('email', teacher?.teacherEmail)

    try {
      const response = await fetch(`${backend_url}/send-final-report`, {
        method: 'POST',
        body: formData
      });
      const data = await response.json();
      setPopupMsg(data.message || 'Report sent successfully');
      setTimeout(() => setPopupMsg(''), 5000);
    } catch (error) {
      setPopupMsg('Error sending report');
      setTimeout(() => setPopupMsg(''), 3000);
    }
  };

  

  // Comments Management
  const addComment = () => {
    if (!newComment.trim()) return;
    
    const commentObj = {
      id: Date.now(),
      text: newComment,
      author: user?.teacherName || 'Teacher',
      role: 'teacher',
      timestamp: new Date().toLocaleString(),
      type: 'general'
    };
    
    setComments([...comments, commentObj]);
    setNewComment('');
  };

  const handleApproveReport = (reportId) => {
    try {
      console.log(comment);
      const formData = new FormData();
      formData.append('reportId', reportId);
      formData.append('comments', comment || '');
      formData.append('teacherId', teacher.teacherId);
      formData.append('status', true);

      fetch(`${backend_url}/teachers/workFlow`, {
        method: 'POST',
        body: formData
      })
      .then(response => response.json())
      .then(data => {
        console.log(data);
        setPopupMsg('Report approved');
        setSelectedReport(null);

        setComment('');
        
        setTimeout(() => setPopupMsg(''), 3000);
      })
      .catch(error => {
        console.error(error);
        setPopupMsg('Error approving report');
        setTimeout(() => setPopupMsg(''), 3000);
      });
    } catch (error) {
      console.error(error);
      setPopupMsg('Error approving report');
      setTimeout(() => setPopupMsg(''), 3000);
    } 
  };

  const handleRejectReport = (reportId) => {
    try {
      console.log(comment);
      const formData = new FormData();
      formData.append('reportId', reportId);
      formData.append('comments', comment || '');
      formData.append('teacherId', teacher.teacherId);
      formData.append('status', false);

      fetch(`${backend_url}/teachers/workFlow`, {
        method: 'POST',
        body: formData
      })
      .then(response => response.json())
      .then(data => {
        console.log(data);
        setPopupMsg('Report rejected successfully');
        setComment('');
        setSelectedReport(null);
        setTimeout(() => setPopupMsg(''), 3000);
      })
      .catch(error => {
        console.error(error);
        setPopupMsg('Error rejecting report');
        setTimeout(() => setPopupMsg(''), 3000);
      });
    } catch (error) {
      console.error(error);
      setPopupMsg('Error rejecting report');
      setTimeout(() => setPopupMsg(''), 3000);
    } 
  };

  const [school, setSchool] = useState({});
  const [teacher, setTeacher] = useState({});
  const [subjects, setSubjects] = useState([]);
  const [grades, setGrades] = useState([]);
  const [scores, setScores] = useState([]);
  const [reportTeacher, setReportTeacher] = useState([]);

  // Fetch data
  useEffect(() => {
    const fetchReports = async (user) => {
      setLoading(true);
      try {
        // 1. First fetch teacher by email
        const teacherRes = await fetch(
          `${backend_url}/teachers/email?email=${user.userEmail}`
        );
        const teacherData = await teacherRes.json();
        setTeacher(teacherData?.data);
  
        // 2. Only proceed if teacher data is valid
        if (!teacherData || !(teacherData?.data).teacherId) {
          return;
        }
        const [studentResponse, workflowResponse, schoolResponse, subjectResponse, gradeResponse, scoreResponse, reportTeacherResponse] = await Promise.all([
          fetch(`${backend_url}/students/id?studentId=${id}`),
          fetch(`${backend_url}/teachers/getWorkflow_all`),
          fetch(`${backend_url}/admin/schools/${(teacherData?.data)?.schoolId}`),
          fetch(`${backend_url}/subjects/`),
          fetch(`${backend_url}/grades/`),
          fetch(`${backend_url}/students/score`),
          fetch(`${backend_url}/teachers/`),

        ]);

        if(!studentResponse.ok || !workflowResponse.ok || !schoolResponse.ok || !subjectResponse.ok || !gradeResponse.ok || !scoreResponse.ok || !reportTeacherResponse.ok) {
          return;
        }

        const studentData = await studentResponse.json();
        const workflowData = await workflowResponse.json();
        const schoolData = await schoolResponse.json();
        const subjectData = await subjectResponse.json();
        const gradeData = await gradeResponse.json();
        const scoreData = await scoreResponse.json();
        const reportTeacher = await reportTeacherResponse.json();

        setSchool(schoolData?.data);
        setSubjects(subjectData?.data);
        setGrades(gradeData?.data);
        setReportTeacher(reportTeacher?.data);

        const filteredScoreData = (scoreData?.data || []).filter(score => score.studentId === Number(id));
        setScores(filteredScoreData);

        const processedData = (workflowData?.data || []).map(item => {
          if (item.report) {
            const base64 = item.report.replace(/\s/g, "");
            const byteCharacters = atob(base64);
            const byteArray = new Uint8Array(byteCharacters.length);
            for (let i = 0; i < byteCharacters.length; i++) {
              byteArray[i] = byteCharacters.charCodeAt(i);
            }
            const blob = new Blob([byteArray], { type: "application/pdf" });
            item.report = URL.createObjectURL(blob);
          }
          return item;
        });

      
    

        setReports(processedData);
        setStudentData(studentData?.data);
      } catch (error) {
        console.error('Error fetching data:', error);
      } finally {
        setLoading(false);
      }
    };
    fetchReports(user);
  }, [user,id]);

  console.log(scores);

  const getGradeColor = (grade) => {
    switch (grade) {
      case 'A': return 'bg-green-100 text-green-800 border border-green-200';
      case 'B': return 'bg-blue-100 text-blue-800 border border-blue-200';
      case 'C': return 'bg-yellow-100 text-yellow-800 border border-yellow-200';
      default: return 'bg-gray-100 text-gray-800 border border-gray-200';
    }
  };

  const getStatusColor = (status) => {
    switch(status) {
      case true: return 'bg-green-100 text-green-800 border border-green-200';
      case false: return 'bg-red-100 text-red-800 border border-red-200';
      default: return 'bg-yellow-100 text-yellow-800 border border-yellow-200';
    }
  };

  const handlePrint = () => {
    if(!signature) {
      setPopupMsg('Please add signature first');
      setTimeout(() => setPopupMsg(''), 3000);
      return;
    }
    window.print();
  }

  if (loading) return <Loader />;

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-4 md:p-8">
      {/* Popup Messages */}
      {popUpMsg && (
        <div className="fixed top-4 right-4 z-100 animate-fade-in">
          <div className="bg-green-500 text-white px-6 py-3 rounded-lg shadow-lg">
            {popUpMsg}
          </div>
        </div>
      )}

      <div className="max-w-6xl mx-auto">
        {/* Action Buttons */}
        <div className="flex flex-wrap gap-3 mb-6 print:hidden">
          <button
            onClick={() => handlePrint()}
            className="flex items-center gap-2 bg-gradient-to-r from-blue-600 to-blue-700 text-white px-6 py-3 rounded-xl hover:from-blue-700 hover:to-blue-800 transition-all duration-300 shadow-lg hover:shadow-xl"
          >
            <Download size={20} />
            Download PDF
          </button>
          
          <button
            onClick={() => setShowSignatureModal(true)}
            className="flex items-center gap-2 bg-gradient-to-r from-indigo-600 to-purple-600 text-white px-6 py-3 rounded-xl hover:from-indigo-700 hover:to-purple-700 transition-all duration-300 shadow-lg hover:shadow-xl"
          >
            <Edit3 size={20} />
            Add Signature
          </button>
          
          <button
            onClick={handleUploadReport}
            className="flex items-center gap-2 bg-gradient-to-r from-green-600 to-emerald-600 text-white px-6 py-3 rounded-xl hover:from-green-700 hover:to-emerald-700 transition-all duration-300 shadow-lg hover:shadow-xl"
          >
            <Mail size={20} />
            Send Report
          </button>
          
          <button
            onClick={() => setShowReportsList(!showReportsList)}
            className="flex items-center gap-2 bg-gradient-to-r from-gray-700 to-gray-800 text-white px-6 py-3 rounded-xl hover:from-gray-800 hover:to-gray-900 transition-all duration-300 shadow-lg hover:shadow-xl"
          >
            <FileText size={20} />
            <span className="flex items-center gap-2">
              View All Reports
              <span className="bg-white text-gray-800 text-xs font-bold px-2 py-1 rounded-full">
                {reports.length}
              </span>
            </span>
          </button>
        </div>

        <div className="flex gap-6">
          {/* Main Report Card */}
          <div className="flex-1">
            <div ref={reportRef} className="bg-white rounded-3xl shadow-2xl overflow-hidden">
              {/* Header */}
              <div className="bg-gradient-to-r from-blue-600 to-indigo-700 text-white p-8 relative overflow-hidden">
                <div className="absolute top-0 right-0 w-80 h-80 bg-white opacity-10 rounded-full -mr-40 -mt-40"></div>
                <div className="flex flex-col md:flex-row justify-between items-center gap-6 relative z-10">
                  <img src={logo} alt="logo" className="w-40 h-auto" />
                  <div className="text-center md:text-right">
                    <h1 className="text-3xl md:text-4xl font-bold mb-2">{school.schoolName}</h1>
                    <p className="text-blue-100 text-sm md:text-base max-w-2xl">
                      {
                        school.address
                      }, {school.city}, {school.state} - {school.pin}
                    </p>
                    <div className="flex flex-wrap justify-center md:justify-end gap-4 mt-4">
                      <div className="flex items-center gap-2">
                        <Phone size={16} />
                        <span>{school.primaryContactNo}</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <Mail size={16} />
                        <a href="mailto:AIS@gmail.com" className="hover:underline">{school.schoolEmail}</a>
                      </div>
                      <div className="flex items-center gap-2">
                        <Globe size={16} />
                        <a href="http://www.aisahmedabad.com" className="hover:underline">www.{school?.schoolName ? (school.schoolName).split(' ').join('').toLowerCase() : ''}.com</a>
                      </div>
                    </div>
                  </div>
                </div>
                <h2 className="text-3xl font-bold text-center mt-8 text-white">Final Report Card</h2>
              </div>

              {/* Student Info */}
              <div className="image p-8 border-b border-gray-200">
                <div className="flex flex-col md:flex-row justify-between items-start gap-6">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4 flex-1">
                    <div className="space-y-3">
                      <InfoRow label="Student Name" value={studentData.studentName || 'N/A'} />
                      <InfoRow label="Roll ID" value={studentData.rollId || 'N/A'} />
                      <InfoRow label="Class" value={studentData.classId || 'N/A'} />
                      <InfoRow label="Head Teacher" value={teacher?.teacherName || 'Teacher Name'} />
                    </div>
                  </div>
                  {studentData.photo && (
                    <div className="ml-8">
                      <img 
                        src={`data:image/jpeg;base64,${studentData.photo}`} 
                        alt="student" 
                        className="w-32 h-32 rounded-2xl object-cover border-4 border-white shadow-lg"
                      />
                    </div>
                  )}
                </div>
              </div>

              {/* Subjects Table */}
              <div className="p-8">
                <h3 className="text-xl font-bold text-gray-800 mb-6">Academic Performance</h3>
                <div className="overflow-hidden rounded-xl border border-gray-200 shadow-sm">
                  <table className="w-full">
                    <thead>
                      <tr className="bg-gradient-to-r from-blue-500 to-blue-600 text-white">
                        <th className="px-6 py-4 text-left font-semibold">Subject</th>
                        <th className="px-6 py-4 text-center font-semibold">Score</th>
                        <th className="px-6 py-4 text-center font-semibold">Grade</th>
                      </tr>
                    </thead>
                    <tbody>
                      {scores.map((subject, index) => (
                        <tr 
                          key={index} 
                          className={`transition-colors ${index % 2 === 0 ? 'bg-gray-50' : 'bg-white'} hover:bg-blue-50`}
                        >
                          <td className="px-6 py-4 font-medium text-gray-900">{subjects.find(s => s.subjectId === subject.subjectId)?.subjectName}</td>
                          <td className="px-6 py-4 text-center font-semibold text-gray-700">{subject.score}</td>
                          <td className="px-6 py-4 text-center">
                            <span className={`inline-block px-4 py-2 rounded-full font-bold ${getGradeColor(grades.find(g => g.gradeId === subject.grade)?.gradeLetter)}`}>
                              {grades.find(g => g.gradeId === subject.grade)?.gradeLetter}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
              

              {/* Final Stats */}
              <div className="px-8 pb-8">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <StatCard 
                    title="Final Score" 
                    value={scores.reduce((acc, curr) => acc + curr.score, 0)} 
                    color="from-blue-500 to-blue-600"
                  />
                  <StatCard 
                    title="Percentage" 
                    value={`${((scores.reduce((acc, curr) => acc + curr.score, 0) / (100 * scores.length)) * 100).toFixed(2)}%`} 
                    color="from-indigo-500 to-purple-600"
                  />
                  <StatCard 
                    title="Percentile Rank" 
                    value={((scores.reduce((acc, curr) => acc + curr.score, 0) / (100 * scores.length)) * 100).toFixed(2)} 
                    color="from-purple-500 to-pink-600"
                  />
                </div>
              </div>

              {/* Signature */}
              <div className="px-8 pb-8">
                <div className="border-t-2 border-gray-300 pt-8 flex justify-end">
                  <div className="text-center">
                    <div className="mb-2">
                      {signature ? (
                        <img 
                          src={signature} 
                          alt="Signature" 
                          className="h-16 max-w-[200px] mx-auto border-2 border-gray-300 rounded-lg"
                        />
                      ) : (
                        <div className="h-16 w-48 border-2 border-dashed border-gray-300 rounded-lg flex items-center justify-center text-gray-400 text-sm bg-gray-50">
                          No signature added
                        </div>
                      )}
                    </div>
                    <div className="border-t-2 border-gray-800 pt-2 w-48">
                      <div className="font-bold text-gray-900">Teacher's Signature</div>
                      <div className="text-sm text-gray-600">{teacher
                      ? teacher.teacherName : 'Head Teacher'}</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Comments Sidebar */}
          {/* <div className="hidden lg:block w-80 print:hidden">
            <div className="bg-white rounded-3xl shadow-2xl p-6 h-fit sticky top-8">
              <h3 className="font-bold text-gray-900 text-lg mb-4 flex items-center gap-2">
                <MessageSquare size={20} />
                Comments & Notes
              </h3>
              
              <div className="space-y-4 mb-6 max-h-96 overflow-y-auto">
                {comments.map((comment) => (
                  <div key={comment.id} className="bg-gray-50 rounded-xl p-4">
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <User size={16} className="text-gray-500" />
                        <span className="font-medium text-gray-800">{comment.author}</span>
                        <span className="text-xs px-2 py-1 bg-blue-100 text-blue-800 rounded-full">
                          {comment.role}
                        </span>
                      </div>
                      <span className="text-xs text-gray-500">{comment.timestamp}</span>
                    </div>
                    <p className="text-gray-700">{comment.text}</p>
                  </div>
                ))}
              </div>

              <div>
                <textarea
                  value={newComment}
                  onChange={(e) => setNewComment(e.target.value)}
                  placeholder="Add a comment..."
                  className="w-full p-4 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent mb-3"
                  rows="3"
                />
                <button
                  onClick={addComment}
                  className="w-full flex items-center justify-center gap-2 bg-gradient-to-r from-blue-600 to-indigo-600 text-white px-6 py-3 rounded-xl hover:from-blue-700 hover:to-indigo-700 transition-all duration-300"
                >
                  <Send size={16} />
                  Add Comment
                </button>
              </div>
            </div>
          </div> */}
        </div>
      </div>

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
                  onChange={handleSignatureUpload}
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

      {/* Reports List Modal */}
      {showReportsList && (
        <Modal
          title="Submitted Reports"
          onClose={() => setShowReportsList(false)}
          width="max-w-4xl"
        >
          <div className="space-y-3 max-h-[60vh] overflow-y-auto">
            {reports.map(report => (
              <div
                key={report.reviewId}
                onClick={() => {
                  handleReportClick(report);
                  setShowReportsList(false);
                }}
                className="flex items-center justify-between p-4 border border-gray-200 rounded-xl hover:bg-gray-50 cursor-pointer transition-all hover:shadow-md"
              >
                <div className="flex-1">
                  <div className="flex items-center gap-3 mb-2">
                    <h3 className="font-bold text-gray-800">{subjects.find(s => s.subjectId === report.subjectId)?.subjectName}</h3>
                    <span className={`px-3 py-1 rounded-full text-xs font-medium ${getStatusColor(report.status)}`}>
                      {report.status === true
                      ? 'Approved' : 'Pending' }
                    </span>
                  </div>
                  <p className="text-sm text-gray-600">Teacher: {reportTeacher.find(t => t.teacherId === report.teacherId)?.teacherName}</p>
                  <p className="text-xs text-gray-500 flex items-center gap-1">
                    {/* <Calendar size={12} /> */}
                    Message: {report.comments}
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  {report.comments && (
                    <MessageSquare size={18} className="text-blue-600" />
                  )}
                  <ChevronRight size={18} className="text-gray-400" />
                </div>
              </div>
            ))}
          </div>
        </Modal>
      )}

      {/* Report Detail Modal */}
      {selectedReport && (
        <Modal
          title={`${selectedReport.comments} Report`}
          onClose={() => setSelectedReport(null)}
          width="max-w-5xl"
        >
          <div className="space-y-6">
            {/* Zoom Controls */}
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-4">
                <button
                  onClick={() => setZoomLevel(Math.max(50, zoomLevel - 10))}
                  className="flex items-center gap-2 px-4 py-2 bg-gray-100 hover:bg-gray-200 rounded-lg transition"
                >
                  <ZoomOut size={18} />
                  Zoom Out
                </button>
                <span className="font-medium bg-gray-100 px-4 py-2 rounded-lg">
                  {zoomLevel}%
                </span>
                <button
                  onClick={() => setZoomLevel(Math.min(200, zoomLevel + 10))}
                  className="flex items-center gap-2 px-4 py-2 bg-gray-100 hover:bg-gray-200 rounded-lg transition"
                >
                  <ZoomIn size={18} />
                  Zoom In
                </button>
              </div>
              
              <div className={`px-4 py-2 rounded-lg ${getStatusColor(selectedReport.status)}`}>
                Status: <span className="font-bold">{selectedReport.status ? 'Approved' : 'Pending'}</span>
              </div>
            </div>

            {/* PDF Viewer */}
            <div className="border-2 border-gray-300 rounded-xl overflow-hidden bg-gray-50">
              <div style={{ transform: `scale(${zoomLevel / 100})`, transformOrigin: 'top left' }}>
                <MyPdfViewer pdfBlobUrl={selectedReport.report} />
              </div>
            </div>

            {/* Comments Section */}
            <div className="space-y-4">
              <h4 className="font-bold text-gray-900 flex items-center gap-2">
                <MessageSquare size={20} />
                Review Comments
              </h4>
              
              {selectedReport.comments && (
                <div className="bg-blue-50 border border-blue-200 rounded-xl p-4">
                  <p className="text-sm font-medium text-gray-700 mb-1">Current Comments:</p>
                  <p className="text-gray-700 italic">"{selectedReport.comments}"</p>
                </div>
              )}

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  {selectedReport.status === 'pending' 
                    ? 'Add your comments (required for rejection, optional for approval)'
                    : 'Add additional comments'}
                </label>
                <textarea
                  onChange={(e) => setComment(e.target.value)}
                  placeholder="Enter your comments here..."
                  className="w-full p-4 border border-gray-300 rounded-xl focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                  rows="4"
                />
              </div>

              {!selectedReport.iStatus && (
                <div className="flex gap-4">
                  <button
                    onClick={() => handleApproveReport(selectedReport.reportId)}
                    className="flex-1 flex items-center justify-center gap-2 px-6 py-3 bg-gradient-to-r from-green-600 to-emerald-600 text-white rounded-xl hover:from-green-700 hover:to-emerald-700 transition"
                  >
                    <CheckCircle size={20} />
                    Approve Report
                  </button>
                  <button
                    onClick={() => handleRejectReport(selectedReport.reportId)}
                    className="flex-1 flex items-center justify-center gap-2 px-6 py-3 bg-gradient-to-r from-red-600 to-pink-600 text-white rounded-xl hover:from-red-700 hover:to-pink-700 transition"
                  >
                    <XCircle size={20} />
                    Reject Report
                  </button>
                </div>
              )}
            </div>
          </div>
        </Modal>
      )}

      <style>{`
        @media print {
          body { margin: 0; }
          .print\\:hidden { display: none !important; }
          .image : {display: flex, justify-content: center; align-items: center;}
        }
        @keyframes fade-in {
          from { opacity: 0; transform: translateY(-10px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .animate-fade-in {
          animation: fade-in 0.3s ease-out;
        }
      `}</style>
    </div>
  );
};

// Helper Components
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

const InfoRow = ({ label, value }) => (
  <div className="flex items-center gap-3">
    <span className="text-gray-600 font-medium min-w-[120px]">{label}:</span>
    <span className="text-gray-900 font-semibold">{value}</span>
  </div>
);

const StatCard = ({ title, value, color }) => (
  <div className={`bg-gradient-to-br ${color} rounded-2xl p-6 text-white shadow-lg`}>
    <div className="text-sm opacity-90 mb-2">{title}</div>
    <div className="text-3xl font-bold">{value}</div>
  </div>
);

export default StudentReportCard;