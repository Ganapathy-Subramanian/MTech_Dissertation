import json
import os
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from typing import Dict, List, Any
import pandas as pd

class AnalyticsDashboard:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.analytics_file = os.path.join(self.base_dir, "crm_analytics.json")
        self.customer_profiles_file = os.path.join(self.base_dir, "customer_profiles.json")

        # Initialize files if they don't exist
        self._init_files()

    def _init_files(self):
        """Initialize analytics and customer profile files"""
        if not os.path.exists(self.analytics_file):
            with open(self.analytics_file, 'w') as f:
                json.dump({"tickets": [], "resolutions": []}, f)

        if not os.path.exists(self.customer_profiles_file):
            with open(self.customer_profiles_file, 'w') as f:
                json.dump({}, f)

    def log_ticket(self, ticket_data):
        """Log incoming ticket for analytics"""
        # Handle both dict and object formats
        if isinstance(ticket_data, dict):
            ticket_entry = {
                "timestamp": datetime.now().isoformat(),
                "text": ticket_data.get("text", ""),
                "customer_id": ticket_data.get("customer_id"),
                "channel": ticket_data.get("channel", "web"),
                "priority": ticket_data.get("priority")
            }
        else:
            ticket_entry = {
                "timestamp": datetime.now().isoformat(),
                "text": ticket_data.text,
                "customer_id": ticket_data.customer_id,
                "channel": ticket_data.channel,
                "priority": ticket_data.priority
            }

        self._append_to_file(self.analytics_file, "tickets", ticket_entry)

    def log_resolution(self, resolution_data):
        """Log ticket resolution for analytics"""
        resolution_entry = {
            "timestamp": datetime.now().isoformat(),
            "source": resolution_data.get("source"),
            "category": resolution_data.get("category"),
            "confidence": resolution_data.get("confidence"),
            "sentiment": resolution_data.get("sentiment"),
            "priority": resolution_data.get("priority"),
            "response_time": self._calculate_response_time(resolution_data)
        }

        self._append_to_file(self.analytics_file, "resolutions", resolution_entry)

    def _append_to_file(self, file_path: str, key: str, data: Dict):
        """Append data to JSON file"""
        try:
            with open(file_path, 'r') as f:
                content = json.load(f)

            content[key].append(data)

            # Keep only last 10000 entries to prevent file from growing too large
            if len(content[key]) > 10000:
                content[key] = content[key][-5000:]

            with open(file_path, 'w') as f:
                json.dump(content, f, indent=2)
        except Exception as e:
            print(f"Error logging to {file_path}: {e}")

    def _calculate_response_time(self, resolution_data) -> float:
        """Calculate response time in seconds (mock implementation)"""
        # In real implementation, this would track actual timing
        source = resolution_data.get("source", "")
        if "RAG" in source:
            return 0.05  # 50ms for memory lookup
        elif "Triage" in source:
            return 0.15  # 150ms for local model
        else:
            return 2.5   # 2.5s for LLM calls

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard analytics"""
        try:
            with open(self.analytics_file, 'r') as f:
                data = json.load(f)

            tickets = data.get("tickets", [])
            resolutions = data.get("resolutions", [])

            # Convert to DataFrames for analysis
            tickets_df = pd.DataFrame(tickets)
            resolutions_df = pd.DataFrame(resolutions)

            if not tickets_df.empty:
                tickets_df['timestamp'] = pd.to_datetime(tickets_df['timestamp'])
            if not resolutions_df.empty:
                resolutions_df['timestamp'] = pd.to_datetime(resolutions_df['timestamp'])

            return {
                "summary": self._get_summary_stats(tickets_df, resolutions_df),
                "charts": self._get_chart_data(tickets_df, resolutions_df),
                "performance": self._get_performance_metrics(resolutions_df),
                "customer_insights": self._get_customer_insights(tickets_df)
            }
        except Exception as e:
            print(f"Error getting dashboard data: {e}")
            return {"error": str(e)}

    def _get_summary_stats(self, tickets_df: pd.DataFrame, resolutions_df: pd.DataFrame) -> Dict:
        """Get summary statistics"""
        now = datetime.now()
        today = now.date()

        # Ensure numeric columns
        if not resolutions_df.empty:
            resolutions_df['response_time'] = pd.to_numeric(resolutions_df['response_time'], errors='coerce')

        avg_response = 0
        if not resolutions_df.empty:
            avg_response = float(resolutions_df['response_time'].mean())
            if pd.isna(avg_response):
                avg_response = 0

        resolution_rate = 0
        if len(tickets_df) > 0:
            resolution_rate = len(resolutions_df) / len(tickets_df)

        escalation_rate = 0
        if not resolutions_df.empty and 'source' in resolutions_df.columns:
            escalation_count = len(resolutions_df[resolutions_df['source'].astype(str).str.contains('LLM', na=False)])
            escalation_rate = escalation_count / len(resolutions_df)

        return {
            "total_tickets_today": len(tickets_df[tickets_df['timestamp'].dt.date == today]) if not tickets_df.empty else 0,
            "total_tickets_week": len(tickets_df[tickets_df['timestamp'] >= now - timedelta(days=7)]) if not tickets_df.empty else 0,
            "avg_response_time": round(avg_response, 2),
            "resolution_rate": round(float(resolution_rate), 2),
            "escalation_rate": round(float(escalation_rate), 2)
        }

    def _get_chart_data(self, tickets_df: pd.DataFrame, resolutions_df: pd.DataFrame) -> Dict:
        """Get data for charts"""
        charts = {}

        # Category distribution
        if not resolutions_df.empty and 'category' in resolutions_df.columns:
            try:
                category_counts = resolutions_df['category'].value_counts().head(10)
                charts['category_distribution'] = {
                    'labels': category_counts.index.tolist(),
                    'data': [int(x) for x in category_counts.values.tolist()]
                }
            except Exception as e:
                print(f"Error in category distribution: {e}")

        # Resolution source distribution
        if not resolutions_df.empty and 'source' in resolutions_df.columns:
            try:
                source_counts = resolutions_df['source'].value_counts()
                charts['resolution_sources'] = {
                    'labels': source_counts.index.tolist(),
                    'data': [int(x) for x in source_counts.values.tolist()]
                }
            except Exception as e:
                print(f"Error in source distribution: {e}")

        # Hourly ticket volume (last 24 hours)
        if not tickets_df.empty:
            try:
                hourly_data = tickets_df.set_index('timestamp').resample('H').size().tail(24)
                charts['hourly_volume'] = {
                    'labels': [f"{i}:00" for i in range(24)],
                    'data': [int(x) for x in hourly_data.tolist()]
                }
            except Exception as e:
                print(f"Error in hourly volume: {e}")

        return charts

    def _get_performance_metrics(self, resolutions_df: pd.DataFrame) -> Dict:
        """Get performance metrics"""
        if resolutions_df.empty:
            return {}

        # Ensure numeric columns
        resolutions_df['response_time'] = pd.to_numeric(resolutions_df['response_time'], errors='coerce')
        resolutions_df['confidence'] = pd.to_numeric(resolutions_df['confidence'], errors='coerce')

        # Calculate average confidence
        avg_confidence = 0
        if 'confidence' in resolutions_df.columns:
            avg_conf = resolutions_df['confidence'].mean()
            if not pd.isna(avg_conf):
                avg_confidence = round(float(avg_conf), 2)

        # Response time by source
        response_time_by_source = {}
        if 'source' in resolutions_df.columns:
            by_source = resolutions_df.groupby('source')['response_time'].mean()
            response_time_by_source = {str(k): round(float(v), 2) if not pd.isna(v) else 0 for k, v in by_source.items()}

        return {
            "avg_confidence": avg_confidence,
            "response_time_by_source": response_time_by_source,
            "resolution_trends": self._calculate_trends(resolutions_df)
        }

    def _calculate_trends(self, df: pd.DataFrame) -> Dict:
        """Calculate resolution trends"""
        if df.empty:
            return {}

        # Ensure response_time and confidence are numeric
        df['response_time'] = pd.to_numeric(df['response_time'], errors='coerce')
        df['confidence'] = pd.to_numeric(df['confidence'], errors='coerce')

        # Group by day and source
        daily_stats = df.set_index('timestamp').resample('D').agg({
            'response_time': 'mean',
            'confidence': 'mean'
        }).tail(7)

        # Convert NaN to 0 for JSON serialization
        response_time_trend = [0 if pd.isna(x) else round(float(x), 2) for x in daily_stats['response_time'].tolist()]
        confidence_trend = [0 if pd.isna(x) else round(float(x), 2) for x in daily_stats['confidence'].tolist()]

        return {
            "response_time_trend": response_time_trend,
            "confidence_trend": confidence_trend
        }

    def _get_customer_insights(self, tickets_df: pd.DataFrame) -> Dict:
        """Get customer behavior insights"""
        if tickets_df.empty:
            return {}

        try:
            # Most active customers
            if 'customer_id' in tickets_df.columns:
                customer_counts = tickets_df['customer_id'].value_counts().head(5)
                most_active_customers = customer_counts.index.tolist()
            else:
                most_active_customers = []

            # Channel distribution
            channel_dist = {}
            if 'channel' in tickets_df.columns:
                channel_dist = {str(k): int(v) for k, v in tickets_df['channel'].value_counts().to_dict().items()}

            return {
                "most_active_customers": most_active_customers,
                "channel_distribution": channel_dist,
                "customer_satisfaction_trend": []  # Would need sentiment data
            }
        except Exception as e:
            print(f"Error in customer insights: {e}")
            return {
                "most_active_customers": [],
                "channel_distribution": {},
                "customer_satisfaction_trend": []
            }

    def get_customer_context(self, customer_id: str) -> str:
        """Get customer context for personalized responses"""
        try:
            with open(self.customer_profiles_file, 'r') as f:
                profiles = json.load(f)

            profile = profiles.get(customer_id, {})
            if profile:
                history = profile.get('history', [])
                recent_tickets = history[-3:]  # Last 3 interactions

                context = f"Customer {profile.get('name', customer_id)} has had {len(history)} previous interactions. "
                if recent_tickets:
                    context += f"Recent issues: {', '.join([t.get('category', 'unknown') for t in recent_tickets])}"

                return context
        except Exception as e:
            print(f"Error getting customer context: {e}")

        return ""

    def update_customer_profile(self, customer_id: str, profile_data: Dict):
        """Update customer profile"""
        try:
            with open(self.customer_profiles_file, 'r') as f:
                profiles = json.load(f)

            profiles[customer_id] = profile_data

            with open(self.customer_profiles_file, 'w') as f:
                json.dump(profiles, f, indent=2)
        except Exception as e:
            print(f"Error updating customer profile: {e}")