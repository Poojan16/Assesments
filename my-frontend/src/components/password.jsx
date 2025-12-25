import React, { useState, useEffect } from 'react';

const PasswordStrengthIndicator = ({ password }) => {
  const [strength, setStrength] = useState(0); // 0: No password, 1: Weak, 2: Moderate, 3: Strong

  useEffect(() => {
    const calculateStrength = () => {
      let newStrength = 0;
      if (!password) {
        setStrength(0);
        return;
      }

      // Criteria for strength
      const hasLowerCase = /[a-z]/.test(password);
      const hasUpperCase = /[A-Z]/.test(password);
      const hasNumber = /\d/.test(password);
      const hasSpecialChar = /[!@#$%^&*(),.?":{}|<>]/.test(password);
      const isLongEnough = password.length >= 8;

      if (isLongEnough) newStrength++;
      if (hasLowerCase) newStrength++;
      if (hasUpperCase) newStrength++;
      if (hasNumber) newStrength++;
      if (hasSpecialChar) newStrength++;

      // Map criteria count to strength level
      if (newStrength <= 2) {
        setStrength(1); // Weak
      } else if (newStrength <= 4) {
        setStrength(2); // Moderate
      } else {
        setStrength(3); // Strong
      }
    };

    calculateStrength();
  }, [password]);

  const getStrengthBarClasses = () => {
    switch (strength) {
      case 1: // Weak
        return 'bg-red-500 w-1/3';
      case 2: // Moderate
        return 'bg-yellow-500 w-2/3';
      case 3: // Strong
        return 'bg-green-500 w-full';
      default: // No password
        return 'bg-gray-300 w-0';
    }
  };

  const getStrengthText = () => {
    switch (strength) {
      case 1:
        return 'Weak';
      case 2:
        return 'Moderate';
      case 3:
        return 'Strong';
      default:
        return '';
    }
  };

  return (
    <div className="w-full mt-2">
      {strength > 0 && (
        <p className={`text-sm mt-1 ${
          strength === 1 ? 'text-red-500' :
          strength === 2 ? 'text-yellow-500' :
          'text-green-500'
        }`}>
          Password Strength: {getStrengthText()}
        </p>
      )}
    </div>
  );
};

export default PasswordStrengthIndicator;