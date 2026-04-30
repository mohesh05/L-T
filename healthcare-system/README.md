# Healthcare Appointment & Booking System

## Folder Structure
- backend (Spring Boot, Security, JWT, JPA/MySQL)
- frontend (Angular scaffold + API service + guards)
- schema.sql

## Run Backend
1. Set `DB_USERNAME` and `DB_PASSWORD`
2. `cd backend && mvn spring-boot:run`

## Run Frontend
1. `cd frontend`
2. `npm install`
3. `ng serve`

## Postman Samples
- POST `/api/auth/register`
- POST `/api/auth/login`
- POST `/api/admin/doctors`
- GET `/api/patient/doctors`
- POST `/api/patient/appointments`
- PATCH `/api/doctor/appointments/{id}/status`

## Screens
- Login/Register page
- Admin dashboard (doctor management)
- Patient dashboard (doctor list, booking, my appointments)
- Doctor dashboard (appointments and status updates)
