package com.healthcare.service.impl;
import com.healthcare.dto.AuthDtos.*;import com.healthcare.entity.*;import com.healthcare.enums.AppointmentStatus;import com.healthcare.exception.ApiException;import com.healthcare.repository.*;import com.healthcare.security.JwtService;import lombok.RequiredArgsConstructor;import org.springframework.security.authentication.*;import org.springframework.security.crypto.password.PasswordEncoder;import org.springframework.stereotype.Service;import java.util.List;
@Service @RequiredArgsConstructor public class AppService {
 private final UserRepository ur; private final DoctorRepository dr; private final AppointmentRepository ar; private final PasswordEncoder pe; private final AuthenticationManager am; private final JwtService jwt;
 public AuthRes register(RegisterReq r){ if(ur.existsByEmail(r.email)) throw new ApiException("Email already exists"); User u=new User(null,r.username,r.email,pe.encode(r.password),r.role); ur.save(u); return new AuthRes(jwt.generate(u.getEmail(),u.getRole().name()),u.getRole().name()); }
 public AuthRes login(LoginReq r){ am.authenticate(new UsernamePasswordAuthenticationToken(r.email,r.password)); User u=ur.findByEmail(r.email).orElseThrow(); return new AuthRes(jwt.generate(u.getEmail(),u.getRole().name()),u.getRole().name()); }
 public Doctor addDoctor(DoctorReq r){ return dr.save(new Doctor(null,r.name,r.specialization,r.availability)); }
 public Appointment book(String email, AppointmentReq r){ User u=ur.findByEmail(email).orElseThrow(); Doctor d=dr.findById(r.doctorId).orElseThrow(()->new ApiException("Doctor not found")); if(ar.existsByDoctorIdAndDate(d.getId(),r.date)) throw new ApiException("Appointment slot already booked"); return ar.save(new Appointment(null,r.date,AppointmentStatus.BOOKED,u,d)); }
 public List<Appointment> patientAppointments(String email){ return ar.findByUserId(ur.findByEmail(email).orElseThrow().getId()); }
 public List<Appointment> doctorAppointments(Long doctorId){ return ar.findByDoctorId(doctorId); }
 public Appointment updateStatus(Long id, StatusReq req){ Appointment a=ar.findById(id).orElseThrow(()->new ApiException("Appointment not found")); a.setStatus(req.status); return ar.save(a); }
 public List<Doctor> allDoctors(){ return dr.findAll(); }
}
