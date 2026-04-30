package com.healthcare.controller;
import com.healthcare.dto.AuthDtos.AppointmentReq;import com.healthcare.entity.*;import com.healthcare.service.impl.AppService;import jakarta.validation.Valid;import lombok.RequiredArgsConstructor;import org.springframework.security.core.annotation.AuthenticationPrincipal;import org.springframework.security.core.userdetails.UserDetails;import org.springframework.web.bind.annotation.*;import java.util.*;
@RestController @RequestMapping("/api/patient") @RequiredArgsConstructor public class PatientController { private final AppService s;
@GetMapping("/doctors") public List<Doctor> doctors(){return s.allDoctors();}
@PostMapping("/appointments") public Appointment book(@AuthenticationPrincipal UserDetails u,@Valid @RequestBody AppointmentReq r){ return s.book(u.getUsername(),r);} 
@GetMapping("/appointments") public List<Appointment> mine(@AuthenticationPrincipal UserDetails u){ return s.patientAppointments(u.getUsername()); }}
