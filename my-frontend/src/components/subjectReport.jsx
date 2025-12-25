import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Phone, Mail, Globe, CheckCircle, AlertCircle } from 'lucide-react';
import { jsPDF } from 'jspdf';
import html2canvas from 'html2canvas';

const BatchStudentReportCard = ({ 
  schoolDetail, 
  studentsData, // Array of student objects
  onBatchComplete 
}) => {
  const [currentStudentIndex, setCurrentStudentIndex] = useState(0);
  const [processingStatus, setProcessingStatus] = useState({
    total: 0,
    processed: 0,
    failed: 0,
    completed: false,
    pdfs: [] // Store all generated PDFs
  });
  const [currentStudent, setCurrentStudent] = useState(null);
  const reportRef = useRef(null);

  console.log(studentsData);
  // Initialize processing
  useEffect(() => {
    if (studentsData && studentsData.length > 0) {
      setProcessingStatus(prev => ({
        ...prev,
        total: studentsData.length
      }));
      setCurrentStudent(studentsData[0]);
    }
  }, [studentsData]);

  // Generate PDF for current student
  useEffect(() => {
    const generateCurrentStudentPDF = async () => {
      if (!currentStudent || !schoolDetail || !reportRef.current) return;

      const studentWithScores = {
        ...currentStudent,
        subjectScores: currentStudent.subjectScores || []
      };

      try {
        // Wait for DOM to render
        await new Promise(resolve => setTimeout(resolve, 300));
        
        // Render component
        const canvas = await html2canvas(reportRef.current, {
          scale: 2,
          useCORS: true,
          logging: false,
          backgroundColor: '#ffffff'
        });

        const imgWidth = 210;
        const imgHeight = (canvas.height * imgWidth) / canvas.width;
        
        const pdf = new jsPDF('p', 'mm', 'a4');
        pdf.addImage(canvas, 'PNG', 0, 0, imgWidth, imgHeight);
        
        const pdfBlob = pdf.output('blob');
        const pdfUrl = URL.createObjectURL(pdfBlob);

        // Update status
        setProcessingStatus(prev => ({
          ...prev,
          processed: prev.processed + 1,
          pdfs: [...prev.pdfs, {
            studentId: currentStudent.studentId || currentStudent.rollId,
            studentName: currentStudent.studentName,
            blob: pdfBlob,
            url: pdfUrl,
            fileName: `${currentStudent.studentName}_Report_${Date.now()}.pdf`
          }]
        }));

        // Move to next student or complete
        if (currentStudentIndex < studentsData.length - 1) {
          setTimeout(() => {
            setCurrentStudentIndex(prev => prev + 1);
            setCurrentStudent(studentsData[currentStudentIndex + 1]);
          }, 500); // Small delay between students
        } else {
          setProcessingStatus(prev => ({ ...prev, completed: true }));
          if (onBatchComplete) {
            console.log('All PDFs generated successfully');
            onBatchComplete({
              total: studentsData.length,
              successful: processingStatus.pdfs.length + 1,
              failed: processingStatus.failed,
              allPdfs: [...processingStatus.pdfs, {
                studentId: currentStudent.studentId || currentStudent.rollId,
                studentName: currentStudent.studentName,
                blob: pdfBlob,
                url: pdfUrl,
                fileName: `${currentStudent.studentName}_Report_${Date.now()}.pdf`
              }]
            });
          }
        }

      } catch (error) {
        console.error(`Failed to generate PDF for ${currentStudent.studentName}:`, error);
        setProcessingStatus(prev => ({
          ...prev,
          failed: prev.failed + 1
        }));

        // Move to next student even if failed
        if (currentStudentIndex < studentsData.length - 1) {
          setCurrentStudentIndex(prev => prev + 1);
          setCurrentStudent(studentsData[currentStudentIndex + 1]);
        } else {
          setProcessingStatus(prev => ({ ...prev, completed: true }));
        }
      }
    };

    if (currentStudent) {
      generateCurrentStudentPDF();
    }
  }, [currentStudent, currentStudentIndex, schoolDetail, studentsData, onBatchComplete]);

  // Get grade color (same as before)
  const getGradeColor = (grade) => {
    const colors = {
      'A': 'bg-green-100 text-green-700 border border-green-300',
      'B': 'bg-blue-100 text-blue-700 border border-blue-300',
      'C': 'bg-yellow-100 text-yellow-700 border border-yellow-300',
      'D': 'bg-orange-100 text-orange-700 border border-orange-300',
      'F': 'bg-red-100 text-red-700 border border-red-300'
    };
    return colors[grade] || 'bg-gray-100 text-gray-700 border border-gray-300';
  };

  if (!currentStudent) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <AlertCircle className="h-12 w-12 text-gray-400 mx-auto mb-4" />
          <p className="text-gray-600">No student data available</p>
        </div>
      </div>
    );
  }

  // Calculate scores for current student
  const subjectScores = currentStudent.subjectScores || [];
  const totalScore = subjectScores.reduce((sum, subject) => sum + (subject.score || 0), 0);
  const maxScore = subjectScores.reduce((sum, subject) => sum + (subject.maxScore || 100), 0);
  const percentage = maxScore > 0 ? ((totalScore / maxScore) * 100).toFixed(2) : 0;
  const percentileRank = Math.min(99, Math.max(1, Math.round(percentage)));

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50 p-4">
      {/* Progress Bar */}
      <div className="max-w-4xl mx-auto mb-6">
        <div className="bg-white rounded-lg shadow p-4">
          <div className="flex justify-between items-center mb-2">
            <h3 className="font-bold text-gray-800">
              Processing Reports: {currentStudent.studentName}
            </h3>
            <span className="text-sm text-gray-600">
              {processingStatus.processed + 1} of {processingStatus.total}
            </span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-4">
            <div 
              className="bg-green-600 h-4 rounded-full transition-all duration-300"
              style={{ width: `${((processingStatus.processed + 1) / processingStatus.total) * 100}%` }}
            ></div>
          </div>
          <div className="mt-2 flex justify-between text-sm text-gray-600">
            <span>✓ {processingStatus.processed} completed</span>
            <span>✗ {processingStatus.failed} failed</span>
            <span>⏳ {processingStatus.total - processingStatus.processed - 1} remaining</span>
          </div>
        </div>
      </div>

      {/* Report Card */}
      <div className="max-w-5xl mx-auto">
        <div ref={reportRef} className="bg-white rounded-xl shadow-2xl overflow-hidden">
          {/* Header */}
          <div className="bg-gradient-to-r from-blue-700 via-indigo-700 to-purple-700 text-white px-8 py-10">
            <div className="flex justify-between items-start mb-6">
              <div className="bg-white/10 backdrop-blur-sm rounded-lg p-3">
                <div className="text-4xl font-bold">🎓</div>
              </div>
              <div className="text-right">
                <h1 className="text-3xl font-bold mb-2">{schoolDetail.schoolName}</h1>
                <p className="text-blue-100 text-sm">
                  {schoolDetail.address}, {schoolDetail.city}, {schoolDetail.state} - {schoolDetail.pin}
                </p>
                <div className="flex flex-wrap justify-end gap-4 mt-3 text-sm">
                  <div className="flex items-center gap-1">
                    <Phone size={14} />
                    <span>{schoolDetail.primaryContactNo}</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <Mail size={14} />
                    <span>{schoolDetail.schoolEmail}</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <Globe size={14} />
                    <span>www.{schoolDetail.schoolName.toLowerCase().replace(/\s+/g, '')}.com</span>
                  </div>
                </div>
              </div>
            </div>
            <div className="text-center border-t-2 border-white/20 pt-6">
              <h2 className="text-3xl font-bold tracking-wide">ACADEMIC REPORT CARD</h2>
            </div>
          </div>

          {/* Student Information */}
          <div className="grid grid-cols-2 gap-6 p-8 bg-gradient-to-r from-blue-50 to-indigo-50 border-b-4 border-blue-200">
            <div className="space-y-3">
              <div className="flex items-center gap-3">
                <span className="text-gray-600 font-semibold min-w-[140px]">Student Name:</span>
                <span className="text-gray-900 font-bold text-lg">{currentStudent.studentName}</span>
              </div>
              <div className="flex items-center gap-3">
                <span className="text-gray-600 font-semibold min-w-[140px]">Roll Number:</span>
                <span className="text-gray-900 font-semibold">{currentStudent.rollId}</span>
              </div>
              <div className="flex items-center gap-3">
                <span className="text-gray-600 font-semibold min-w-[140px]">Class/Section:</span>
                <span className="text-gray-900 font-semibold">{currentStudent.classId}</span>
              </div>
            </div>
            <div className="space-y-3">
              <div className="flex items-center gap-3">
                <span className="text-gray-600 font-semibold min-w-[140px]">Class Teacher:</span>
                <span className="text-gray-900 font-semibold">{currentStudent.headTeacher}</span>
              </div>
              <div className="flex items-center gap-3">
                <span className="text-gray-600 font-semibold min-w-[140px]">Date of Issue:</span>
                <span className="text-gray-900 font-semibold">{new Date().toLocaleDateString()}</span>
              </div>
            </div>
          </div>

          {/* Academic Performance */}
          <div className="p-8">
            <h3 className="text-2xl font-bold text-gray-800 mb-6 flex items-center gap-2">
              <span className="text-3xl">📊</span>
              Academic Performance
            </h3>
            <div className="overflow-hidden rounded-xl border-2 border-gray-200 shadow-lg">
              <table className="w-full">
                <thead>
                  <tr className="bg-gradient-to-r from-blue-600 to-indigo-600 text-white">
                    <th className="px-6 py-4 text-left font-bold text-sm uppercase tracking-wide">Subject</th>
                    <th className="px-6 py-4 text-center font-bold text-sm uppercase tracking-wide">Max Score</th>
                    <th className="px-6 py-4 text-center font-bold text-sm uppercase tracking-wide">Obtained</th>
                    <th className="px-6 py-4 text-center font-bold text-sm uppercase tracking-wide">Grade</th>
                  </tr>
                </thead>
                <tbody>
                  {subjectScores.map((subject, index) => (
                    <tr 
                      key={index} 
                      className={`${index % 2 === 0 ? 'bg-gray-50' : 'bg-white'} hover:bg-blue-50`}
                    >
                      <td className="px-6 py-4 font-semibold text-gray-900">{subject.subjectName}</td>
                      <td className="px-6 py-4 text-center text-gray-700 font-medium">{subject.maxScore || 100}</td>
                      <td className="px-6 py-4 text-center font-bold text-gray-900 text-lg">{subject.score}</td>
                      <td className="px-6 py-4 text-center">
                        <span className={`inline-block px-4 py-2 rounded-lg font-bold text-sm ${getGradeColor(subject.grade)}`}>
                          {subject.grade}
                        </span>
                      </td>
                    </tr>
                  ))}
                  <tr className="bg-gradient-to-r from-indigo-100 to-blue-100 border-t-4 border-indigo-300">
                    <td className="px-6 py-5 font-bold text-gray-900 text-lg">TOTAL</td>
                    <td className="px-6 py-5 text-center font-bold text-gray-900 text-lg">{maxScore}</td>
                    <td className="px-6 py-5 text-center font-bold text-indigo-700 text-xl">{totalScore}</td>
                    <td className="px-6 py-5"></td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          {/* Performance Summary */}
          <div className="px-8 pb-8">
            <div className="grid grid-cols-3 gap-6">
              <div className="bg-gradient-to-br from-emerald-500 to-teal-600 rounded-xl p-6 text-white shadow-xl">
                <div className="text-sm font-medium opacity-90 mb-2">Total Score</div>
                <div className="text-4xl font-bold">{totalScore}</div>
                <div className="text-sm opacity-75 mt-1">out of {maxScore}</div>
              </div>
              <div className="bg-gradient-to-br from-blue-500 to-indigo-600 rounded-xl p-6 text-white shadow-xl">
                <div className="text-sm font-medium opacity-90 mb-2">Percentage</div>
                <div className="text-4xl font-bold">{percentage}%</div>
                <div className="text-sm opacity-75 mt-1">Overall Performance</div>
              </div>
              <div className="bg-gradient-to-br from-purple-500 to-pink-600 rounded-xl p-6 text-white shadow-xl">
                <div className="text-sm font-medium opacity-90 mb-2">Percentile Rank</div>
                <div className="text-4xl font-bold">{percentileRank}th</div>
                <div className="text-sm opacity-75 mt-1">Class Position</div>
              </div>
            </div>
          </div>

          {/* Comments Section */}
          {currentStudent.comments && (
            <div className="px-8 pb-8">
              <div className="bg-amber-50 border-l-4 border-amber-500 rounded-lg p-6">
                <h3 className="font-bold text-gray-900 mb-3 text-lg flex items-center gap-2">
                  <span>💬</span> Teacher's Comments
                </h3>
                <p className="text-gray-700 leading-relaxed italic">{currentStudent.comments}</p>
              </div>
            </div>
          )}

          {/* Signature Section */}
          <div className="px-8 pb-8 border-t-2 border-gray-200 pt-8">
            <div className="flex justify-end items-end">
              <div className="text-center">
                <div className="w-48 border-b-2 border-gray-800 mb-2 pb-16"></div>
                <div className="font-semibold text-gray-900">Class Teacher's Signature</div>
                <div className="text-sm text-gray-600 mt-1">{currentStudent.headTeacher}</div>
              </div>
            </div>
          </div>

          {/* Footer */}
          <div className="bg-gradient-to-r from-blue-700 to-indigo-700 text-white text-center py-4">
            <p className="text-sm">This is a computer-generated document and requires no signature</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default BatchStudentReportCard;