from tamev import Choice, Noul, Score

REAL_CASES = [
    {
        "domain": "Fintech & Banking",
        "name": "Disputed Card Transaction / Travel Emergency",
        "state": (
            "Customer message: 'I am currently in Tokyo and my Visa debit card was blocked after "
            "attempting to withdraw ¥50,000 at a 7-Eleven ATM. I did not notify the bank of travel beforehand. "
            "I have zero local currency and need immediate access to funds to check into my hotel.'"
        ),
        "questions": {
            "routing": Choice(
                instructions="Which banking service queue should handle this case?",
                criteria={
                    "travel_unblock": "Verify identity and immediately lift international security travel block",
                    "dispute_charge": "Open an unauthorized transaction fraud dispute claim",
                    "pin_reset": "Send temporary PIN code reset instructions",
                    "branch_visit": "Direct customer to visit physical local branch during banking hours",
                },
            ),
            "is_emergency": Noul(
                instructions="Is this customer in an urgent emergency stranded situation?",
                criteria={
                    "false": "Routine account inquiry",
                    "true": "Stranded customer without funds abroad",
                },
            ),
            "escalation_score": Score(
                instructions="Rate customer urgency and escalation level",
                criteria=[
                    "Level 0: Informational query, standard 24h response queue",
                    "Level 1: Non-urgent account change, 8h response queue",
                    "Level 2: Time-sensitive travel inconvenience, 1h response queue",
                    "Level 3: Critical emergency, immediate live specialist transfer",
                ],
            ),
        },
    },
    {
        "domain": "E-Commerce & Logistics",
        "name": "Perishable Shipment Delivery Failure",
        "state": {
            "order_id": "ORD-2026-9812",
            "item": "Specialty Refrigerated Insulin Medication (Keep at 2-8°C)",
            "shipping_method": "Cold-Chain Next Day Air",
            "issue": "Carrier marked 'Delivered on front porch' at 1:15 PM in 95°F heat; recipient not home until 6:00 PM",
            "customer_note": "Package is sitting in direct sunlight. Medication will spoil if temperature exceeds 8°C for over 2 hours!",
        },
        "questions": {
            "remediation_action": Choice(
                instructions="Select immediate logistics remediation action",
                criteria={
                    "emergency_reship": "Dispatch emergency replacement from nearest regional pharmacy hub via courier",
                    "wait_and_inspect": "Advise customer to inspect package upon arrival home and verify thermometer strip",
                    "refund_only": "Issue full refund to customer credit card",
                    "carrier_claim": "File standard delivery investigation claim with postal carrier",
                },
            ),
            "critical_risk": Noul(
                instructions="Is product at risk of irreversible spoiling?",
                criteria={
                    "false": "Resilient goods with no thermal sensitivity",
                    "true": "Perishable medication in high ambient temperature",
                },
            ),
            "severity_tier": Score(
                instructions="Rate complaint severity tier",
                criteria=[
                    "Tier 0: Cosmetic packaging damage",
                    "Tier 1: Minor delivery delay with ambient-stable contents",
                    "Tier 2: High-value lost non-perishable merchandise",
                    "Tier 3: Compromised medical supplies / health-critical failure",
                ],
            ),
        },
    },
    {
        "domain": "DevOps & Cloud SRE",
        "name": "Kubernetes Production Database Outage",
        "state": {
            "alert": "Firing: CriticalProductionServiceOutage",
            "cluster": "prod-us-east-1",
            "component": "patroni-postgres-leader",
            "event": "OOMKilled exit code 137; connection pool saturated at 10,000/10,000",
            "impact": "Core checkout and billing API returning HTTP 503 Service Unavailable; error rate 94.2%",
            "active_users": 18500,
        },
        "questions": {
            "mitigation_playbook": Choice(
                instructions="Select automated SRE incident mitigation playbook",
                criteria={
                    "failover_standby": "Promote warm standby replica to primary leader and trigger connection pool drain",
                    "restart_container": "Restart current OOM pod in-place with existing memory limits",
                    "throttle_traffic": "Apply aggressive API rate-limiting to all non-essential read endpoints",
                    "page_leads": "Send escalation page to senior engineering leads without automated intervention",
                },
            ),
            "customer_outage": Noul(
                instructions="Is this a customer-facing production outage?",
                criteria={
                    "false": "Internal staging or batch test failure",
                    "true": "High-impact live customer service disruption",
                },
            ),
            "incident_severity": Score(
                instructions="Classify incident severity tier",
                criteria=[
                    "SEV-4: Minor internal tool degradation",
                    "SEV-3: Non-critical feature impacted with available workaround",
                    "SEV-2: Major product feature degraded for subset of users",
                    "SEV-1: Critical customer-facing outage across primary revenue flow",
                ],
            ),
        },
    },
    {
        "domain": "Healthcare Clinical Triage",
        "name": "Emergency Department Inbound Assessment",
        "state": (
            "Patient: 62-year-old female presenting with sudden-onset acute unilateral facial droop, "
            "right arm weakness (pronator drift positive), and slurred speech starting 40 minutes prior to arrival. "
            "Blood pressure 178/102 mmHg, blood glucose 110 mg/dL. Prior medical history: Atrial Fibrillation."
        ),
        "questions": {
            "triage_pathway": Choice(
                instructions="Determine immediate clinical diagnostic pathway",
                criteria={
                    "code_stroke": "Activate Code Stroke protocol: stat non-contrast head CT and neurology consult within 15 min",
                    "urgent_care": "Refer to outpatient urgent care clinic for routine neurological evaluation",
                    "bedside_vitals": "Repeat vital signs in triage waiting area every 30 minutes",
                    "blood_work_only": "Order routine comprehensive metabolic panel and observe",
                },
            ),
            "time_sensitive": Noul(
                instructions="Is this a hyperacute time-sensitive thrombolytic window emergency?",
                criteria={
                    "false": "Chronic or non-urgent symptom progression",
                    "true": "Acute onset within thrombolytic/thrombectomy window (< 4.5h)",
                },
            ),
            "acuity_score": Score(
                instructions="Assign Emergency Severity Index (ESI) acuity rating",
                criteria=[
                    "ESI-5: Non-urgent (prescription refill)",
                    "ESI-4: Less urgent (simple rash, suture removal)",
                    "ESI-3: Urgent (moderate pain, stable vital signs)",
                    "ESI-2: Emergent (high risk, confused/lethargic, severe pain)",
                    "ESI-1: Resuscitation (immediate life-saving intervention needed)",
                ],
            ),
        },
    },
    {
        "domain": "Smart Home & IoT Safety",
        "name": "Unattended Gas Range & Thermal Anomaly",
        "state": {
            "sensors": {
                "kitchen_stove": "Burner 3 active for 85 minutes at high heat",
                "flame_sensor": "Unstable thermal radiation detected",
                "smoke_optical": "0.12 dB/m (smoke concentration rising)",
                "pir_motion_home": "0 events in past 50 minutes (house unoccupied)",
                "smart_lock": "Locked from outside, geolocation shows resident 4 miles away",
            }
        },
        "questions": {
            "safety_action": Choice(
                instructions="Select automated smart home protective intervention",
                criteria={
                    "shutoff_gas_valve": "Actuate emergency motorized shutoff valve to isolate gas supply",
                    "send_push_notification": "Send standard battery notification to user app",
                    "sound_chime": "Sound friendly doorbell chime",
                    "ignore": "Log telemetry reading and take no action",
                },
            ),
            "hazard_present": Noul(
                instructions="Is there an unattended fire or explosion hazard?",
                criteria={
                    "false": "Normal attended household cooking",
                    "true": "Unattended high-temperature fire hazard",
                },
            ),
            "danger_scale": Score(
                instructions="Evaluate safety hazard scale",
                criteria=[
                    "Scale 0: Normal safe operation",
                    "Scale 1: Minor anomaly, self-limiting",
                    "Scale 2: Moderate safety concern requiring resident alert",
                    "Scale 3: Critical imminent hazard requiring immediate shutoff",
                ],
            ),
        },
    },
]
