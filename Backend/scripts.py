# import pandas as pd
# import random
# from datetime import date, timedelta

# # Name lists
# male_names = [
#     "Aarav","Vivaan","Aditya","Vihaan","Arjun","Reyansh","Ayaan"
# ]

# female_names = [
#     "Ananya","Diya","Isha","Kavya","Pooja","Riya","Saanvi",
#     "Meera","Shreya","Shruti"
# ]

# first_names = male_names + female_names

# last_names = [
#     "Sharma","Verma","Gupta","Patel","Singh",
#     "Kumar","Mehta","Iyer","Nair","Reddy"
# ]

# states_cities = {
#     "Maharashtra": ["Mumbai", "Pune", "Nagpur", "Thane", "Solapur", "Aurangabad"],
#     "Karnataka": ["Bengaluru", "Mysuru", "Hubli", "Mangalore", "Belgaum"],
#     "Delhi": ["New Delhi", "North Delhi", "South Delhi", "East Delhi", "West Delhi"],
#     "Tamil Nadu": ["Chennai", "Coimbatore", "Salem", "Madurai", "Trichy"],
#     "Uttar Pradesh": ["Lucknow", "Noida", "Ghaziabad", "Varanasi", "Allahabad"],
#     "Gujarat": ["Ahmedabad", "Surat", "Vadodara", "Rajkot", "Baroda"],
#     "Rajasthan": ["Jaipur", "Jodhpur", "Udaipur", "Ajmer", "Bikaner"],
#     "Kerala": ["Thiruvananthapuram", "Kozhikode", "Kochi", "Kollam", "Alappuzha"],
#     "Odisha": ["Bhubaneswar", "Cuttack", "Puri", "Rourkela", "Baleshwar"],
#     "Punjab": ["Chandigarh", "Ludhiana", "Amritsar", "Jalandhar"],
#     "Andhra Pradesh": ["Vijayawada", "Vizag", "Guntur", "Kakinada", "Rajahmundry"],
# }

# grades = ["N"]

# def random_dob(class_no):
#     age = 5 + class_no
#     start = date.today().replace(year=date.today().year - age - 1)
#     end = date.today().replace(year=date.today().year - age)
#     return start + timedelta(days=random.randint(0, (end - start).days))

# rows = []
# unique_keys = set()

# for school_id in range(1, 11):          # 10 schools
#     for class_no in range(1, 6):        # Class 1–5

#         count = 0
#         while count < 30:               # 30 students per class

#             fn = random.choice(first_names)

#             if fn in male_names:
#                 gender = "male"
#                 parent_fn = random.choice(male_names)
#             else:
#                 gender = "female"
#                 parent_fn = random.choice(female_names)

#             ln = random.choice(last_names)
#             state = random.choice(list(states_cities.keys()))
#             city = random.choice(states_cities[state])
#             dob = random_dob(class_no)

#             # Unique key definition
#             unique_key = (fn, ln, dob, class_no, school_id)

#             if unique_key in unique_keys:
#                 continue  # regenerate

#             unique_keys.add(unique_key)

#             rows.append({
#                 "Student Name": f"{fn} {ln}",
#                 "DOB": dob,
#                 "Gender": gender,
#                 "Class": class_no,
#                 "Address": f"House No {random.randint(1, 250)}",
#                 "City": city,
#                 "State": state,
#                 "Country": "India",
#                 "Pincode": random.randint(100000, 999999),
#                 "Grade": random.choice(grades),
#                 "Parent Name": f"{parent_fn} {ln}",
#                 "Parent Email": f"{parent_fn.lower()}{ln.lower()}{random.randint(1,999)}@gmail.com",
#                 "SchoolId": school_id
#             })

#             count += 1

# df = pd.DataFrame(rows)

# # Final safety check
# assert df.duplicated().sum() == 0, "Duplicate records found!"

# df.to_excel("indian_students_data.xlsx", index=False)

# print(f"Total Students Generated: {len(df)}")
# print("Excel file created: indian_students_data.xlsx")



# import pandas as pd
# import random
# from datetime import date, timedelta

# # Separate male & female names
# male_names = ["Amit", "Rahul", "Suresh", "Ravi", "Kiran", "Vijay", "Rajesh", "Rahul", "Raj", "Ramesh"]
# female_names = ["Anita", "Sunita", "Priya", "Neha", "Pooja", "Kavita", "Kiran", "Komal", "Sneha", "Nisha"]
# last_names = ["Sharma", "Verma", "Patel", "Iyer", "Reddy", "Singh", "Gupta", "Mehta", "Joshi", "Kumar", "Rao"]

# cities_states = [
#     ("Mumbai", "Maharashtra"),
#     ("Delhi", "Delhi"),
#     ("Bengaluru", "Karnataka"),
#     ("Chennai", "Tamil Nadu"),
#     ("Hyderabad", "Telangana"),
#     ("Pune", "Maharashtra"),
#     ("Ahmedabad", "Gujarat"),
#     ("Kolkata", "West Bengal"),
#     ("Jaipur", "Rajasthan"),
#     ("Bhopal", "Madhya Pradesh"),
# ]

# qualifications = ["B.Ed", "M.Ed", "B.Sc B.Ed", "M.Sc B.Ed"]

# records = []
# teacher_id = 1

# for school_id in range(1, 11):     # 10 schools
#     for _ in range(5):             # 5 teachers per school

#         if random.choice([True, False]):
#             fname = random.choice(male_names)
#             gender = "male"
#         else:
#             fname = random.choice(female_names)
#             gender = "female"

#         lname = random.choice(last_names)
#         city, state = random.choice(cities_states)

#         dob = date.today() - timedelta(days=random.randint(25*365, 45*365))
#         onboarding_date = date.today() - timedelta(days=random.randint(30, 1500))

#         records.append({
#             "teacherName": f"{fname} {lname}",
#             "teacherEmail": f"{fname.lower()}.{lname.lower()}{teacher_id}@school.com",
#             "teacherContact": f"9{random.randint(100000000, 999999999)}",
#             "emergencyContact": f"9{random.randint(100000000, 999999999)}",
#             "onboardingDate": onboarding_date,
#             "address": f"House {random.randint(1,200)}, {city}",
#             "city": city,
#             "state": state,
#             "country": "India",
#             "pin": random.randint(110000, 560000),
#             "qualification": random.choice(qualifications),
#             "role": "Teacher",
#             "schoolId": school_id,
#             "DOB": dob,
#             "gender": gender
#         })

#         teacher_id += 1

# # Export to Excel
# df = pd.DataFrame(records)
# df.to_excel("teacher_data_india.xlsx", index=False)

# print("Excel file generated: teacher_data_india.xlsx")

