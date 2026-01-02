import React, { useEffect, useState } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { initializeAuth } from '../authSlice';

const TeacherMapping = () => {
  const { user } = useSelector((state) => state.auth);
  const dispatch = useDispatch();

  useEffect(() => {
    dispatch(initializeAuth());
  }, [dispatch]);

  const [teachers, setTeachers] = useState([]);
  const [classes, setClasses] = useState([]);
  const [subjects, setSubjects] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [successMessage, setSuccessMessage] = useState('');
  const [errorMessage, setErrorMessage] = useState('');
  const [selectedTeacherIndex, setSelectedTeacherIndex] = useState(0);
  
  // Search functionality state
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [isSearching, setIsSearching] = useState(false);
  const [showDropdown, setShowDropdown] = useState(false);
  const [teacher, setTeacher] = useState({});
  const backend_url = process.env.REACT_APP_BACKEND_URL;
  useEffect(() => {
    const fetchTeacher = async (user) => {
      try {
        const response = await fetch(`${backend_url}/teachers/email?email=` + user.userEmail);
        const data = await response.json();
        setTeacher(data?.data);
      } catch (error) {
        console.log(error);
      }
    };
    fetchTeacher(user);
  }, [user]);

  useEffect(() => {
    const fetchData = async () => {
      setIsLoading(true);
      try {
        const [classesRes, subjectsRes] = await Promise.all([
          fetch(`${backend_url}/classes/`),
          fetch(`${backend_url}/subjects/`)
        ]);
        const classesData = await classesRes.json();
        const subjectsData = await subjectsRes.json();
        const filteredClasses = (classesData?.data).filter(c => c.schoolId === 1);
        setClasses(filteredClasses);
        setSubjects(subjectsData?.data);
      } catch (error) {
        setErrorMessage('Failed to load data');
        console.error(error);
      } finally {
        setIsLoading(false);
      }
    };
    fetchData();
  }, []);

  // Search teachers by email
  const searchTeachers = async (query,teacher) => {
    if (query && !query.trim()) {
      setSearchResults([]);
      setShowDropdown(false);
      return;
    }

    setIsSearching(true);
    console.log(teacher)
    try {
      const response = await fetch(`${backend_url}/teachers/search?query=${encodeURIComponent(query)}&schoolId=${teacher?.schoolId}`);
      const data = await response.json();
      setSearchResults(data?.data || []);
      setShowDropdown(true);
    } catch (error) {
      console.error('Search failed:', error);
      setSearchResults([]);
    } finally {
      setIsSearching(false);
    }
  };

  // Debounced search
  useEffect(() => {
    const delaySearch = setTimeout(() => {
      searchTeachers(searchQuery,teacher);
    }, 300);

    return () => clearTimeout(delaySearch);
  }, [searchQuery,teacher]);

  const addTeacher = () => {
    setTeachers([...teachers, { email: '', name: '', mappings: {} }]);
    setSelectedTeacherIndex(teachers.length);
    setSearchQuery('');
  };

  const selectTeacherFromSearch = (teacher) => {
    const newTeachers = [...teachers];
    newTeachers[selectedTeacherIndex].email = teacher.teacherEmail;
    newTeachers[selectedTeacherIndex].name = teacher.teacherName;
    setTeachers(newTeachers);
    setSearchQuery(teacher.email);
    setShowDropdown(false);
  };

  const removeTeacher = (index) => {
    const newTeachers = teachers.filter((_, i) => i !== index);
    setTeachers(newTeachers);
    if (selectedTeacherIndex >= newTeachers.length) {
      setSelectedTeacherIndex(Math.max(0, newTeachers.length - 1));
    }
  };

  const toggleMapping = (teacherIndex, classId, subjectId) => {
    const newTeachers = [...teachers];
    const mappings = newTeachers[teacherIndex].mappings;
    if (!mappings[classId]) {
      mappings[classId] = [];
    }
    const subjectIndex = mappings[classId].indexOf(subjectId);
    if (subjectIndex > -1) {
      mappings[classId].splice(subjectIndex, 1);
      if (mappings[classId].length === 0) {
        delete mappings[classId];
      }
    } else {
      mappings[classId].push(subjectId);
    }
    setTeachers(newTeachers);
  };

  const toggleAllSubjects = (teacherIndex, classId, subjectIds) => {
    const newTeachers = [...teachers];
    const mappings = newTeachers[teacherIndex].mappings;
    const allSelected = subjectIds.every(sid => mappings[classId]?.includes(sid));
    if (allSelected) {
      delete mappings[classId];
    } else {
      mappings[classId] = [...subjectIds];
    }
    setTeachers(newTeachers);
  };

  const handleSubmit = async () => {
    if (teachers.length === 0) {
      setErrorMessage('Please add at least one teacher');
      return;
    }

    const formattedData = {
      teachers: teachers.map(t => ({
        email: t.email,
        classes: Object.entries(t.mappings).map(([classId, subjectIds]) => ({
          classId: classId,
          subjects: subjectIds
        }))
      }))
    };

    try {
      const response = await fetch(`${backend_url}/teachers/map_teacher`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formattedData),
      });

      if (response.ok) {
        setSuccessMessage('Teacher mapped successfully');
        setErrorMessage('');
        setTimeout(() => {
          setSuccessMessage('');
          setTeachers([]);
        }, 3000);
      } else {
        setErrorMessage('Teacher mapping failed');
        setSuccessMessage('');
        setTimeout(() => setErrorMessage(''), 3000);
      }
    } catch (error) {
      setErrorMessage('Failed to submit mapping');
      console.error(error);
    }
  };

  const getClassSubjects = () => {
    return subjects;
  };

  const currentTeacher = teachers[selectedTeacherIndex];

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-xl text-gray-600">Loading...</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 p-4 md:p-8">
      <div className="max-w-6xl mx-auto">
        <h1 className="text-3xl font-bold text-gray-800 mb-6">Teacher Class & Subject Mapping</h1>

        {successMessage && (
          <div className="mb-4 p-4 bg-green-100 border border-green-400 text-green-700 rounded-md">
            {successMessage}
          </div>
        )}

        {errorMessage && (
          <div className="mb-4 p-4 bg-red-100 border border-red-400 text-red-700 rounded-md">
            {errorMessage}
          </div>
        )}

        <div className="mb-6 flex gap-3">
          <button
            onClick={addTeacher}
            className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 font-medium"
          >
            + Add Teacher
          </button>
          {teachers.length > 0 && (
            <button
              onClick={handleSubmit}
              className="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 font-medium"
            >
              Submit All Mappings
            </button>
          )}
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {teachers.length > 0 && (
            <div className="lg:col-span-1 bg-white rounded-lg shadow p-4">
              <h2 className="text-lg font-semibold mb-4 text-gray-700">Teachers</h2>
              <div className="space-y-2">
                {teachers.map((teacher, index) => (
                  <div
                    key={index}
                    className={`p-3 rounded-md cursor-pointer flex justify-between items-center ${
                      selectedTeacherIndex === index
                        ? 'bg-blue-100 border-2 border-blue-500'
                        : 'bg-gray-50 border border-gray-200 hover:bg-gray-100'
                    }`}
                  >
                    <div onClick={() => setSelectedTeacherIndex(index)} className="flex-1">
                      {teacher.email ? (
                        <>
                          <span className="font-medium text-gray-700 truncate block">
                            {teacher.name || teacher.email.split('@')[0]}
                          </span>
                          <div className="text-xs text-gray-500 mt-1 truncate">{teacher.email}</div>
                        </>
                      ) : (
                        <span className="font-medium text-gray-400 italic">Search teacher...</span>
                      )}
                    </div>
                    <button
                      onClick={() => removeTeacher(index)}
                      className="text-red-500 hover:text-red-700 font-bold text-xl ml-2"
                    >
                      ×
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className={`${teachers.length > 0 ? 'lg:col-span-3' : 'lg:col-span-4'}`}>
            {currentTeacher && (
              <div className="bg-white rounded-lg shadow p-6">
                <div className="mb-6">
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Search Teacher by Email *
                  </label>
                  <div className="relative">
                    <input
                      type="text"
                      value={searchQuery}
                      onChange={(e) => {
                        setSearchQuery(e.target.value);
                        if (e.target.value !== currentTeacher.email) {
                          const newTeachers = [...teachers];
                          newTeachers[selectedTeacherIndex].email = '';
                          setTeachers(newTeachers);
                        }
                      }}
                      onFocus={() => searchQuery && setShowDropdown(true)}
                      onBlur={() => setTimeout(() => setShowDropdown(false), 200)}
                      placeholder="Type to search teacher email..."
                      className="w-full md:w-96 p-3 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                      required
                    />
                    {isSearching && (
                      <div className="absolute right-3 top-3">
                        <div className="animate-spin h-5 w-5 border-2 border-blue-500 border-t-transparent rounded-full"></div>
                      </div>
                    )}
                    
                    {showDropdown && searchResults.length > 0 && (
                      <div className="absolute z-10 w-full md:w-96 mt-1 bg-white border border-gray-300 rounded-md shadow-lg max-h-60 overflow-y-auto">
                        {searchResults.map((result, idx) => (
                          <div
                            key={idx}
                            onClick={() => selectTeacherFromSearch(result)}
                            className="p-3 hover:bg-blue-50 cursor-pointer border-b border-gray-100 last:border-b-0"
                          >
                            <div className="font-medium text-gray-800">{result.teacherName}</div>
                            <div className="text-sm text-gray-500">{result.teacherEmail}</div>
                          </div>
                        ))}
                      </div>
                    )}
                    
                    {showDropdown && searchQuery && !isSearching && searchResults.length === 0 && (
                      <div className="absolute z-10 w-full md:w-96 mt-1 bg-white border border-gray-300 rounded-md shadow-lg p-3">
                        <div className="text-gray-500 text-sm">No teachers found</div>
                      </div>
                    )}
                  </div>
                  {currentTeacher.email && (
                    <div className="mt-2 text-sm text-green-600">
                      ✓ Selected: {currentTeacher.email}
                    </div>
                  )}
                </div>


                <div>
                  <h2 className="text-xl font-semibold text-gray-800 mb-4">
                    Select Classes & Subjects
                  </h2>
                  <div className="overflow-x-auto">
                    <table className="min-w-full bg-white border border-gray-200 rounded-lg">
                      <thead className="bg-gray-100">
                        <tr>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-700 uppercase tracking-wider border-b">
                            Class
                          </th>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-700 uppercase tracking-wider border-b">
                            Subjects
                          </th>
                          <th className="px-6 py-3 text-center text-xs font-medium text-gray-700 uppercase tracking-wider border-b">
                            Select All
                          </th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-gray-200">
                        {classes.map((classItem) => {
                          const classSubjects = getClassSubjects();
                          const allSelected = classSubjects.every(s =>
                            currentTeacher.mappings[classItem.classId]?.includes(s.subjectId)
                          );
                          const someSelected = classSubjects.some(s =>
                            currentTeacher.mappings[classItem.classId]?.includes(s.subjectId)
                          );
                          console.log(classSubjects)

                          return (
                            <tr key={classItem.classId} className="hover:bg-gray-50">
                              <td className="px-6 py-4 whitespace-nowrap font-medium text-gray-900">
                                {classItem.className}
                              </td>
                              <td className="px-6 py-4">
                                <div className="flex flex-wrap gap-3">
                                  {classSubjects.map((subject) => (
                                    <label
                                      key={subject.subjectId}
                                      className="flex items-center space-x-2 cursor-pointer"
                                    >
                                      <input
                                        type="checkbox"
                                        checked={currentTeacher.mappings[classItem.classId]?.includes(subject.subjectId) || false}
                                        onChange={() => toggleMapping(selectedTeacherIndex, classItem.classId, subject.subjectId)}
                                        className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
                                      />
                                      <span className="text-sm text-gray-700">
                                        {subject.subjectName}
                                      </span>
                                    </label>
                                  ))}
                                </div>
                              </td>
                              <td className="px-6 py-4 text-center">
                                <button
                                  onClick={() => toggleAllSubjects(
                                    selectedTeacherIndex,
                                    classItem.classId,
                                    classSubjects.map(s => s.subjectId)
                                  )}
                                  className={`px-3 py-1 rounded text-sm font-medium transition-colors ${
                                    allSelected
                                      ? 'bg-blue-600 text-white hover:bg-blue-700'
                                      : someSelected
                                      ? 'bg-blue-100 text-blue-700 hover:bg-blue-200'
                                      : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                                  }`}
                                >
                                  {allSelected ? 'Deselect All' : 'Select All'}
                                </button>
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            )}

            {teachers.length === 0 && (
              <div className="bg-white rounded-lg shadow p-12 text-center">
                <div className="text-gray-400 text-lg mb-2">No teachers added yet.</div>
                <div className="text-gray-500">Click "Add Teacher" to get started.</div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default TeacherMapping;