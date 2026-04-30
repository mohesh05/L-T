package com.healthcare.entity;
import com.healthcare.enums.AppointmentStatus;import jakarta.persistence.*;import lombok.*;import java.time.LocalDateTime;
@Entity @Getter @Setter @NoArgsConstructor @AllArgsConstructor
public class Appointment { @Id @GeneratedValue(strategy=GenerationType.IDENTITY) private Long id; private LocalDateTime date; @Enumerated(EnumType.STRING) private AppointmentStatus status; @ManyToOne private User user; @ManyToOne private Doctor doctor; }
