<?php

namespace App\Exports;

use Carbon\Carbon;
use App\Models\Sertifikasi;
use Illuminate\Support\Facades\Auth;
use Maatwebsite\Excel\Concerns\WithHeadings;
use Illuminate\Contracts\Support\Responsable;
use Maatwebsite\Excel\Concerns\FromCollection;

class SertifikasiExport implements FromCollection, WithHeadings
{
    protected $request;

    public function __construct($request)
    {
        $this->request = $request;
    }

    public function collection()
    {
        $user = Auth::user();

        $query = Sertifikasi::with(['user.mitraProfile', 'daftarSertifikasis'])
            ->withCount('daftarSertifikasis')
            ->orderBy('created_at', 'desc');

        // Jika bukan admin, filter berdasarkan user_id
        if ($user->role !== 'admin') {
            $query->where('user_id', $user->id);
        }

        // Terapkan semua filter
        if ($this->request->filled('judul')) {
            $query->where('judul_sertifikasi', 'like', '%' . $this->request->judul . '%');
        }

        if ($this->request->filled('kota')) {
            $query->where('kota', 'like', '%' . $this->request->kota . '%');
        }

        if ($this->request->filled('status')) {
            $query->where('status', $this->request->status);
        }

        if ($this->request->filled('tahun_mulai')) {
            $query->whereYear('tanggal_mulai', $this->request->tahun_mulai);
        }

        if ($this->request->filled('status_selesai')) {
            if ($this->request->status_selesai == 'selesai') {
                $query->whereDate('tanggal_selesai', '<', Carbon::today());
            } elseif ($this->request->status_selesai == 'belum') {
                $query->whereDate('tanggal_selesai', '>=', Carbon::today());
            }
        }

        $sertifikasis = $query->get();

        return $sertifikasis->map(function ($s) {
            $pesertas = collect($s->daftarSertifikasis);

            return [
                'Instansi' => $s->user->mitraProfile->nama_instansi ?? '-',
                'Judul Sertifikasi' => $s->judul_sertifikasi ?? '-',
                'Kota' => $s->kota ?? '-',
                'Jumlah Peserta' => $s->daftar_sertifikasis_count ?? 0,
                'Jumlah Asal Sekolah' => $pesertas->pluck('asal_sekolah')->unique()->count() ?? 0,
                'Jumlah Jurusan' => $pesertas->pluck('jurusan')->unique()->count() ?? 0,
                'Jumlah Bekerja/Wirausaha' => $pesertas->whereIn('status_saat_ini', ['Bekerja', 'Wirausaha'])->count() ?? 0,
                'Jumlah Tidak Bekerja' => $pesertas->where('status_saat_ini', 'Tidak Bekerja')->count() ?? 0,
                'Jumlah Kuliah' => $pesertas->where('status_saat_ini', 'Kuliah')->count() ?? 0,
            ];
        });
    }

    public function headings(): array
    {
        return [
            'Instansi',
            'Judul Sertifikasi',
            'Kota',
            'Jumlah Peserta',
            'Jumlah Asal Sekolah',
            'Jumlah Jurusan',
            'Jumlah Bekerja/Wirausaha',
            'Jumlah Tidak Bekerja',
            'Jumlah Kuliah',
        ];
    }
}
