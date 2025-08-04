"""
PDF Generator for Railway Track Anomaly Detection System

Creates detailed PDF reports with annotated images, GPS coordinates, and detection information.
"""

import cv2
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.pdfgen import canvas

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import PDF_CONFIG, DETECTION_CLASSES

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PDFGenerator:
    """Generates PDF reports with detection results"""
    
    def __init__(self):
        self.output_dir = Path(PDF_CONFIG["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize styles
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        
    def _setup_custom_styles(self):
        """Setup custom paragraph styles"""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.darkblue
        ))
        
        # Subtitle style
        self.styles.add(ParagraphStyle(
            name='CustomSubtitle',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceAfter=20,
            textColor=colors.darkgreen
        ))
        
        # Detection style
        self.styles.add(ParagraphStyle(
            name='DetectionInfo',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=10,
            leftIndent=20
        ))
        
        # GPS style
        self.styles.add(ParagraphStyle(
            name='GPSInfo',
            parent=self.styles['Normal'],
            fontSize=9,
            textColor=colors.grey
        ))
        
    def _draw_bounding_box(self, image: cv2.Mat, detection: Dict) -> cv2.Mat:
        """Draw bounding box and label on image"""
        annotated_image = image.copy()
        
        bbox = detection["bbox"]
        x1, y1, x2, y2 = bbox
        
        # Get detection info
        class_name = detection["class"]
        confidence = detection["confidence"]
        severity = detection.get("severity", "MEDIUM")
        
        # Choose color based on severity
        if severity == "HIGH":
            color = (0, 0, 255)  # Red
        elif severity == "MEDIUM":
            color = (0, 165, 255)  # Orange
        else:
            color = (0, 255, 255)  # Yellow
            
        # Draw bounding box
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
        
        # Create label text
        label = f"{class_name.upper()}: {confidence:.2f}"
        
        # Get text size
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        # Draw label background
        cv2.rectangle(annotated_image, 
                     (x1, y1 - text_height - 10), 
                     (x1 + text_width + 10, y1), 
                     color, -1)
        
        # Draw label text
        cv2.putText(annotated_image, label, 
                   (x1 + 5, y1 - 5), 
                   font, font_scale, (255, 255, 255), thickness)
        
        return annotated_image
        
    def _find_annotated_frame(self, frame_number: int, timestamp: str) -> Optional[Path]:
        """Find saved annotated frame by frame number and timestamp"""
        try:
            frames_dir = Path("data/frames")
            if not frames_dir.exists():
                return None
                
            # Look for frame with matching frame number and timestamp
            timestamp_clean = timestamp.replace(':', '-')
            pattern = f"detection_frame_{frame_number:04d}_{timestamp_clean}*.jpg"
            
            for frame_file in frames_dir.glob(pattern):
                if frame_file.exists():
                    return frame_file
                    
            # If not found, try to find any frame with this frame number
            pattern = f"detection_frame_{frame_number:04d}_*.jpg"
            for frame_file in frames_dir.glob(pattern):
                if frame_file.exists():
                    return frame_file
                    
            return None
            
        except Exception as e:
            logger.error(f"Error finding annotated frame: {e}")
            return None
        
    def _create_summary_table(self, detections: List[Dict]) -> Table:
        """Create summary table of all detections"""
        # Count detections by class
        detection_counts = {}
        severity_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
        
        for detection in detections:
            class_name = detection["class"]
            severity = detection.get("severity", "MEDIUM")
            
            detection_counts[class_name] = detection_counts.get(class_name, 0) + 1
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Create table data
        table_data = [
            ["Detection Type", "Count", "Severity", "Action Required"]
        ]
        
        for class_name, count in detection_counts.items():
            class_info = DETECTION_CLASSES.get(class_name, {})
            severity = class_info.get("severity", "MEDIUM")
            action = class_info.get("action_required", "Inspect area")
            
            table_data.append([class_name.replace("_", " ").title(), 
                             str(count), severity, action])
        
        # Create table
        table = Table(table_data, colWidths=[2*inch, 0.8*inch, 1*inch, 2.2*inch])
        
        # Style the table
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        
        return table
        
    def _create_detection_details(self, detection: Dict) -> List:
        """Create detailed information for a single detection"""
        elements = []
        
        # Detection header
        class_name = detection["class"].replace("_", " ").title()
        confidence = detection["confidence"]
        severity = detection.get("severity", "MEDIUM")
        
        header_text = f"{class_name} (Confidence: {confidence:.2f}, Severity: {severity})"
        elements.append(Paragraph(header_text, self.styles["CustomSubtitle"]))
        
        # Detection details
        details = [
            f"<b>Description:</b> {detection.get('description', 'No description available')}",
            f"<b>Action Required:</b> {detection.get('action_required', 'Inspect area')}",
            f"<b>Timestamp:</b> {detection.get('timestamp', 'Unknown')}",
            f"<b>GPS Coordinates:</b> {detection.get('gps', {}).get('lat', 'N/A'):.6f}, {detection.get('gps', {}).get('lng', 'N/A'):.6f}",
            f"<b>Bounding Box:</b> {detection.get('bbox', 'N/A')}"
        ]
        
        for detail in details:
            elements.append(Paragraph(detail, self.styles["DetectionInfo"]))
            
        elements.append(Spacer(1, 20))
        
        return elements
        
    def generate_report(self, frame_data: List[Dict], output_filename: Optional[str] = None) -> str:
        """Generate PDF report from frame data"""
        if not frame_data:
            logger.warning("No frame data provided for report generation")
            return ""
            
        # Generate output filename if not provided
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"railway_detection_report_{timestamp}.pdf"
            
        output_path = self.output_dir / output_filename
        
        # Create PDF document
        doc = SimpleDocTemplate(str(output_path), pagesize=A4)
        story = []
        
        # Title page
        story.append(Paragraph("Railway Track Anomaly Detection Report", self.styles["CustomTitle"]))
        story.append(Spacer(1, 30))
        
        # Report metadata
        report_info = [
            f"<b>Report Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"<b>Total Frames Analyzed:</b> {len(frame_data)}",
            f"<b>Total Detections:</b> {sum(len(frame.get('detections', [])) for frame in frame_data)}",
            f"<b>Drone ID:</b> {frame_data[0].get('drone_id', 'Unknown')}"
        ]
        
        for info in report_info:
            story.append(Paragraph(info, self.styles["Normal"]))
            
        story.append(Spacer(1, 30))
        
        # Summary table
        all_detections = []
        for frame in frame_data:
            all_detections.extend(frame.get('detections', []))
            
        if all_detections:
            story.append(Paragraph("Detection Summary", self.styles["CustomSubtitle"]))
            story.append(self._create_summary_table(all_detections))
            story.append(Spacer(1, 30))
        
        # Process each frame with detections
        frames_with_detections = [frame for frame in frame_data if frame.get('detections')]
        
        for i, frame_info in enumerate(frames_with_detections):
            frame = frame_info['frame']
            detections = frame_info['detections']
            gps = frame_info.get('gps', {})
            timestamp = frame_info.get('timestamp', 'Unknown')
            
            # Frame header
            story.append(Paragraph(f"Frame {i+1} Analysis", self.styles["CustomSubtitle"]))
            
            # GPS and timestamp info
            gps_text = f"GPS: {gps.get('lat', 'N/A'):.6f}, {gps.get('lng', 'N/A'):.6f} | Time: {timestamp}"
            story.append(Paragraph(gps_text, self.styles["GPSInfo"]))
            story.append(Spacer(1, 10))
            
            # Try to find saved annotated frame
            frame_number = frame_info.get('frame_number', i)
            timestamp = frame_info.get('timestamp', 'unknown')
            annotated_frame_path = self._find_annotated_frame(frame_number, timestamp)
            
            if annotated_frame_path and annotated_frame_path.exists():
                # Use saved annotated frame
                img = Image(str(annotated_frame_path), width=6*inch, height=4.5*inch)
                story.append(img)
                story.append(Spacer(1, 10))
            else:
                # Create annotated frame on the fly
                annotated_frame = frame.copy()
                for detection in detections:
                    annotated_frame = self._draw_bounding_box(annotated_frame, detection)
                
                # Save annotated image temporarily
                temp_image_path = Path(f"temp_frame_{i}.jpg")
                cv2.imwrite(str(temp_image_path), annotated_frame)
                
                # Add image to PDF
                img = Image(str(temp_image_path), width=6*inch, height=4.5*inch)
                story.append(img)
                story.append(Spacer(1, 10))
                
                # Clean up temporary file
                temp_image_path.unlink(missing_ok=True)
            
            # Add detection details
            for detection in detections:
                story.extend(self._create_detection_details(detection))
            
            # Add page break if not last frame
            if i < len(frames_with_detections) - 1:
                story.append(Spacer(1, 20))
        
        # Build PDF
        doc.build(story)
        
        logger.info(f"PDF report generated: {output_path}")
        return str(output_path)
        
    def generate_simple_report(self, detections: List[Dict], output_filename: Optional[str] = None) -> str:
        """Generate a simple report from detection data only"""
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"simple_detection_report_{timestamp}.pdf"
            
        output_path = self.output_dir / output_filename
        
        # Create PDF document
        doc = SimpleDocTemplate(str(output_path), pagesize=A4)
        story = []
        
        # Title
        story.append(Paragraph("Railway Track Detection Summary", self.styles["CustomTitle"]))
        story.append(Spacer(1, 30))
        
        # Summary table
        if detections:
            story.append(Paragraph("Detection Summary", self.styles["CustomSubtitle"]))
            story.append(self._create_summary_table(detections))
            story.append(Spacer(1, 30))
            
            # Individual detection details
            story.append(Paragraph("Detection Details", self.styles["CustomSubtitle"]))
            for detection in detections:
                story.extend(self._create_detection_details(detection))
        
        # Build PDF
        doc.build(story)
        
        logger.info(f"Simple PDF report generated: {output_path}")
        return str(output_path)


def main():
    """Test the PDF generator with sample data"""
    import argparse
    
    parser = argparse.ArgumentParser(description="PDF Report Generator")
    parser.add_argument("--output", default="test_report.pdf", help="Output filename")
    
    args = parser.parse_args()
    
    # Create sample data for testing
    sample_detections = [
        {
            "class": "missing_clamp",
            "confidence": 0.95,
            "bbox": [100, 150, 200, 250],
            "description": "Missing rail clamp detected",
            "severity": "HIGH",
            "action_required": "Immediate inspection required",
            "timestamp": datetime.now().isoformat(),
            "gps": {"lat": 40.7128, "lng": -74.0060}
        },
        {
            "class": "debris",
            "confidence": 0.75,
            "bbox": [300, 200, 350, 250],
            "description": "Debris on track detected",
            "severity": "LOW",
            "action_required": "Clean track",
            "timestamp": datetime.now().isoformat(),
            "gps": {"lat": 40.7129, "lng": -74.0061}
        }
    ]
    
    # Create PDF generator and generate report
    generator = PDFGenerator()
    output_path = generator.generate_simple_report(sample_detections, args.output)
    
    print(f"Report generated: {output_path}")


if __name__ == "__main__":
    main() 