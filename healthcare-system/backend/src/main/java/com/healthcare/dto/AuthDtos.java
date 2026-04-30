package com.healthcare.dto;
import com.healthcare.enums.*;import jakarta.validation.constraints.*;import lombok.Data;import java.time.LocalDateTime;
public class AuthDtos {
 @Data public static class RegisterReq { @NotBlank public String username; @Email public String email; @NotBlank public String password; public Role role=Role.PATIENT; }
 @Data public static class LoginReq { @Email public String email; @NotBlank public String password; }
 @Data public static class AuthRes { public String token; public String role; public AuthRes(String t,String r){token=t;role=r;} }
 @Data public static class DoctorReq{ @NotBlank public String name; @NotBlank public String specialization; @NotBlank public String availability; }
 @Data public static class AppointmentReq{ @NotNull public Long doctorId; @NotNull public LocalDateTime date; }
 @Data public static class StatusReq{ @NotNull public AppointmentStatus status; }
}
