import React, { useState, useEffect } from 'react';
import car from '../assets/car.png';
import hand from '../assets/hand.png';

const directions = ['up', 'upRight', 'right', 'downRight', 'down', 'downLeft','left', 'upLeft'];

const CarRotationActivity = ({ onSubmitSuccess }) => {

  const [carDirection, setCarDirection] = useState('up');
  const [handDirection, setHandDirection] = useState(null); // fixed direction per attempt
  const [attempts, setAttempts] = useState(0);
  const [result, setResult] = useState(null);
  const [disabled, setDisabled] = useState(false);

  // Pick 1 random hand direction at start OR on retry
  useEffect(() => {
    setHandDirection(directions[Math.floor(Math.random() * directions.length)]);
  }, []);

  const rotateLeft = (e) => {
    e.preventDefault();
    if (disabled) return;
    const idx = directions.indexOf(carDirection);
    const newDir = directions[(idx - 1 + directions.length) % directions.length];
    setCarDirection(newDir);
  };

  const rotateRight = (e) => {
    e.preventDefault();
    if (disabled) return;
    const idx = directions.indexOf(carDirection);
    const newDir = directions[(idx + 1) % directions.length];
    setCarDirection(newDir);
  };

  const submitDirection = (e) => {
    e.preventDefault();
    if (disabled) return;

    const match = carDirection === handDirection;
    const newAttempts = attempts + 1;
    setAttempts(newAttempts);

    if (match) {
      setResult("Success! Direction matched.");
      setDisabled(true);
      onSubmitSuccess(true);
      return;
    }

    // Failed attempt
    if (newAttempts >= 2) {
      setResult("Failed! You have used all attempts.");
      setDisabled(true);
      onSubmitSuccess(false);
    } else {
      setResult("Incorrect. Try one more time.");
      // Car resets but hand remains fixed
      setCarDirection("up");
    }
  };

  const angleMap = {
    up: 0,
    upRight: 45,
    right: 90,
    downRight: 135,
    down: 180,
    downLeft: 315,
    left: 270,
    upLeft: 225,
  };

  return (
    <div className='max-w-md mx-auto' style={{ textAlign: 'center', padding: 7 }}>
      <h2><b>Please verify you are not a robot.</b><br/>Rotate the car to match the direction of the hand.</h2>

      <div className="flex" style={{ display: 'flex', justifyContent: 'center' }}>
        
        {/* HAND IMAGE (fixed direction per attempt) */}
        <div className='mr-12'>
          <div style={{
            margin: '20px auto',
            width: 120,
            height: 120,
            transform: `rotate(${handDirection ? angleMap[handDirection]-90 : 0}deg)`,
            transition: '0.3s',
            backgroundImage: `url(${hand})`,
            backgroundColor: 'lightgray',
            backgroundSize: 'contain',
            backgroundRepeat: 'no-repeat',
            border: '1px solid black'
          }} />
          <p>Target Hand Direction</p>
        </div>

        {/* CAR IMAGE */}
        <div>
          <div style={{
            margin: '20px auto',
            width: 120,
            height: 120,
            transform: `rotate(${angleMap[carDirection]}deg)`,
            transition: '0.3s',
            backgroundImage: `url(${car})`,
            backgroundColor: 'lightblue',
            backgroundSize: 'contain',
            backgroundRepeat: 'no-repeat',
            border: '1px solid black',
          }} />

          {/* Controls */}
          <div style={{ marginBottom: 20 }}>
            <button disabled={disabled} onClick={rotateLeft}>⬅ Left</button>
            <button disabled={disabled} onClick={rotateRight} style={{ marginLeft: 10 }}>Right ➡</button>
          </div>
        </div>

      </div>

      <button
        disabled={disabled}
        className='bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded'
        onClick={submitDirection}
      >
        Submit Direction
      </button>

      {result && (
        <div className='mt-4' style={{ fontSize: 15 }}>
          {result}
        </div>
      )}

    </div>
  );
};

export default CarRotationActivity;
