import type { DomainLabel } from "../types";

const names = [
  "Arts_and_Entertainment", "Autos_and_Vehicles", "Beauty_and_Fitness",
  "Books_and_Literature", "Business_and_Industrial", "Computers_and_Electronics",
  "Finance", "Food_and_Drink", "Games", "Health", "Hobbies_and_Leisure",
  "Home_and_Garden", "Internet_and_Telecom", "Jobs_and_Education",
  "Law_and_Government", "News", "Online_Communities", "People_and_Society",
  "Pets_and_Animals", "Real_Estate", "Science", "Sensitive_Subjects",
  "Shopping", "Sports", "Travel_and_Transportation",
] as const;

/** Local reference list for browsing while the API is offline. The API is authoritative. */
export const REFERENCE_DOMAINS: DomainLabel[] = names.map((name, id) => ({ id, name }));

export const VIETNAMESE_DOMAINS: Record<string, string> = {
  Arts_and_Entertainment: "Nghệ thuật & Giải trí",
  Autos_and_Vehicles: "Ô tô & Phương tiện",
  Beauty_and_Fitness: "Làm đẹp & Thể hình",
  Books_and_Literature: "Sách & Văn học",
  Business_and_Industrial: "Kinh doanh & Công nghiệp",
  Computers_and_Electronics: "Máy tính & Điện tử",
  Finance: "Tài chính",
  Food_and_Drink: "Ẩm thực & Đồ uống",
  Games: "Trò chơi",
  Health: "Sức khỏe",
  Hobbies_and_Leisure: "Sở thích & Giải trí",
  Home_and_Garden: "Nhà cửa & Sân vườn",
  Internet_and_Telecom: "Internet & Viễn thông",
  Jobs_and_Education: "Việc làm & Giáo dục",
  Law_and_Government: "Luật pháp & Chính phủ",
  News: "Tin tức",
  Online_Communities: "Cộng đồng trực tuyến",
  People_and_Society: "Con người & Xã hội",
  Pets_and_Animals: "Thú cưng & Động vật",
  Real_Estate: "Bất động sản",
  Science: "Khoa học",
  Sensitive_Subjects: "Chủ đề nhạy cảm",
  Shopping: "Mua sắm",
  Sports: "Thể thao",
  Travel_and_Transportation: "Du lịch & Di chuyển",
};

export function displayDomain(name: string): string {
  return VIETNAMESE_DOMAINS[name] ?? name.replaceAll("_", " ");
}
